import math
import time
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torchvision.models import mobilenet_v3_small, MobileNetV3


def mlp(num_channels: int):
    """blah"""
    return nn.Sequential(
        nn.LayerNorm(num_channels),
        nn.Linear(num_channels, num_channels),
        nn.GELU(),
        nn.Linear(num_channels, num_channels),
    )


def _generate_positions_for_encoding(spatial_shape, v_min=-1.0, v_max=1.0):
    """
    Create evenly spaced position coordinates for
    spatial_shape with values in [v_min, v_max].
    :param v_min: minimum coordinate value per dimension.
    :param v_max: maximum coordinate value per dimension.
    :return: position coordinates tensor of shape (*shape, len(shape)).
    """
    coords = [torch.linspace(v_min, v_max, steps=s) for s in spatial_shape]
    return torch.stack(torch.meshgrid(*coords), dim=len(spatial_shape))


def _generate_position_encodings(
    p: Tensor,
    num_frequency_bands: int,
    max_frequencies: Optional[Tuple[int, ...]] = None,
    include_positions: bool = True,
) -> Tensor:
    """Fourier-encode positions p using num_frequency_bands.
    :param p: positions of shape (*d, c) where c = len(d).
    :param max_frequencies: maximum frequency for each dimension (1-tuple for sequences,
           2-tuple for images, ...). If `None` values are derived from shape of p.
    :param include_positions: whether to include input positions p in returned encodings tensor.
    :returns: position encodings tensor of shape (*d, c * (2 * num_bands + include_positions)).
    """
    encodings = []

    if max_frequencies is None:
        max_frequencies = p.shape[:-1]

    frequencies = [
        torch.linspace(1.0, max_freq / 2.0, num_frequency_bands, device=p.device)
        for max_freq in max_frequencies
    ]
    frequency_grids = []

    for i, frequencies_i in enumerate(frequencies):
        frequency_grids.append(p[..., i : i + 1] * frequencies_i[None, ...])

    if include_positions:
        encodings.append(p)

    encodings.extend(
        [torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids]
    )
    encodings.extend(
        [torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids]
    )

    return torch.cat(encodings, dim=-1)


class TimeDecoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 256, history_length: int = 16
    ) -> None:
        super().__init__()
        self.history_length = history_length

        # Setup actual decoder
        self.value_tf = nn.Linear(input_dim, hidden_dim)
        self.time_attn = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)

        # Setup time position encoding
        p = _generate_positions_for_encoding([history_length])
        time_enc = _generate_position_encodings(
            p, hidden_dim // 2, include_positions=False
        )
        self.register_buffer("time_encoding", time_enc, persistent=False)

        # Setup learned embedding to query time
        self.time_query = nn.Parameter(torch.empty(1, hidden_dim))
        with torch.no_grad():
            self.time_query.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

        # Setup decoding output of mhsa to a yaw estimate
        self.decode_yawrate = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        time_encoding = self.time_encoding.expand([x.shape[0], -1, -1])
        time_query = self.time_query.expand([x.shape[0], -1, -1])
        values = self.value_tf(x)
        yaw_embed, _ = self.time_attn(time_query, time_encoding, values)
        yaw_estimate = self.decode_yawrate(yaw_embed)
        return yaw_estimate


def _forward_impl_patch(self, x: Tensor) -> Tensor:
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    return x


class SequenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = mobilenet_v3_small()
        # moneky patch to not classify
        self.encoder._forward_impl = _forward_impl_patch.__get__(
            self.encoder, MobileNetV3
        )
        self.encoder_dim = self.encoder.classifier[0].in_features
        self.decoder = TimeDecoder(self.encoder_dim)

    def forward(self, x):
        """Image stack [b,t,c,h,w]"""
        time_tokens = []
        for t_idx in range(x.shape[1]):
            time_tokens.append(self.encoder(x[:, t_idx]))
        time_tokens = torch.stack(time_tokens, dim=1)  # [b,t,c]
        x = self.decoder(time_tokens)  # [b,c]
        return x


class SingleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = mobilenet_v3_small()
        # moneky patch to not classify
        self.encoder._forward_impl = _forward_impl_patch.__get__(
            self.encoder, MobileNetV3
        )
        encoder_feats = self.encoder.classifier[0].in_features

        self.decoder = nn.Sequential(
            nn.LayerNorm(encoder_feats),
            nn.Linear(encoder_feats, encoder_feats // 2),
            nn.GELU(),
            nn.Linear(encoder_feats // 2, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_model(config) -> nn.Module:
    models = {
        "SequenceModel": SequenceModel,
        "SingleModel": SingleModel,
    }

    model_cfg = config["model"]
    model = models[model_cfg["type"]](**model_cfg["args"])

    return model


def profile_model(model, dummy, niter: int) -> float:
    model(dummy)  # warm up
    start = time.perf_counter()
    for _ in range(niter):
        model(dummy)
    return (time.perf_counter() - start) / niter


@torch.inference_mode()
def profile_sequence(niter: int) -> None:
    from torch.utils.mobile_optimizer import optimize_for_mobile

    dummy_input = torch.empty((1, 16, 3, 240, 320))
    model = SequenceModel().eval()

    print(f"Std Time Taken {profile_model(model, dummy_input, niter):.3f}s")

    with torch.jit.optimized_execution(True):
        jit_model = torch.jit.trace(model, dummy_input)
        print(f"JIT Time Taken {profile_model(jit_model, dummy_input, niter):.3f}s")
        torch.jit.save(jit_model, "model.jit.pt")

        mob_optim = optimize_for_mobile(jit_model)
        print(f"Mobile Time Taken {profile_model(mob_optim, dummy_input, niter):.3f}s")

        torch.jit.save(mob_optim, "mobile.jit.pt")

    with torch.autocast(device_type="cpu"):
        print(f"Autocast Time Taken {profile_model(model, dummy_input, niter):.3f}s")


def export_onnx() -> None:
    dummy_input = torch.empty((1, 16, 3, 240, 320))
    model = SequenceModel().eval()

    torch.onnx.export(
        model, dummy_input, "model.onnx", opset_version=13, input_names=["input"]
    )


def test_onnx(niter: int) -> None:
    import onnxruntime as ort
    import numpy as np

    dummy_input = np.empty((1, 16, 3, 240, 320))
    session = ort.InferenceSession("model.onnx")
    session.run(None, {"input": dummy_input})  # warm up

    start = time.perf_counter()
    for _ in range(niter):
        session.run(None, {"input": dummy_input})
    time_taken = (time.perf_counter() - start) / niter
    print(f"onnx time taken {time_taken:.2f}")


def export_parts() -> None:
    model = SequenceModel().eval()
    image = torch.empty((1, 3, 240, 320))
    tokens = torch.empty((1, model.decoder.history_length, model.encoder_dim))

    torch.onnx.export(
        model.encoder, image, "encoder.onnx", opset_version=13, input_names=["image"]
    )

    torch.onnx.export(
        model.decoder, tokens, "decoder.onnx", opset_version=13, input_names=["tokens"]
    )


def profile_parts(niter: int) -> None:
    import onnxruntime as ort
    import numpy as np

    # Encode image
    session = ort.InferenceSession("encoder.onnx")
    image = np.empty(session.get_inputs()[0].shape, dtype=np.float32)
    print(f"Image shape: {image.shape}")
    session.run(None, {"image": image})  # warm up

    start = time.perf_counter()
    for _ in range(niter):
        session.run(None, {"image": image})
    time_taken = (time.perf_counter() - start) / niter
    print(f"encoder time taken {time_taken:.2f}")

    # Decode yaw rate from features
    session = ort.InferenceSession("decoder.onnx")
    tokens = np.empty(session.get_inputs()[0].shape, dtype=np.float32)
    print(f"Token shape: {tokens.shape}")
    session.run(None, {"tokens": tokens})  # warm up

    start = time.perf_counter()
    for _ in range(niter):
        session.run(None, {"tokens": tokens})
    time_taken = (time.perf_counter() - start) / niter
    print(f"decoder time taken {time_taken:.2f}")


def test_inference() -> None:
    niter = 5
    # export_parts()
    profile_parts(niter)


if __name__ == "__main__":
    test_inference()
