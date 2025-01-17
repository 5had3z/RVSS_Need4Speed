import math
import time
from pathlib import Path
from functools import partial
from typing import Any, Dict, Optional, Tuple, Literal

import torch
from torch import nn, Tensor
from torchvision.models.mobilenetv3 import (
    mobilenet_v3_small,
    MobileNetV3,
    MobileNet_V3_Small_Weights,
    InvertedResidualConfig,
)


def mobilenet_v3_tiny(
    width_mult: float = 1.0,
    reduced_tail: bool = False,
    dilated: bool = False,
    **kwargs: Any,
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(
        InvertedResidualConfig.adjust_channels, width_mult=width_mult
    )

    # fmt: off
    inverted_residual_setting = [
            bneck_conf(8, 3, 8, 8, True, "RE", 2, 1),  # C1
            bneck_conf(8, 3, 36, 12, False, "RE", 2, 1),  # C2
            bneck_conf(12, 3, 44, 12, False, "RE", 1, 1),
            bneck_conf(12, 5, 48, 20, True, "HS", 2, 1),  # C3
            bneck_conf(20, 5, 120, 20, True, "HS", 1, 1),
            bneck_conf(20, 5, 120, 24, True, "HS", 1, 1),
            bneck_conf(24, 5, 144, 48 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(48 // reduce_divider, 5, 288 // reduce_divider, 48 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(48 // reduce_divider, 5, 288 // reduce_divider, 48 // reduce_divider, True, "HS", 1, dilation),
        ]
    last_channel = adjust_channels(512 // reduce_divider)  # C5
    # fmt: on

    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)

    return model


def mlp(num_channels: int):
    """blah"""
    return nn.Sequential(
        nn.LayerNorm(num_channels),
        nn.Linear(num_channels, num_channels),
        nn.GELU(),
        nn.Linear(num_channels, num_channels),
    )


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

    return torch.cat(encodings, dim=-1).to(torch.float32)


def _generate_embeddings(
    time_offset: Tensor, time_normalize: float, hidden_dim: int
) -> Tensor:
    """Generate feature embedding for time offsets"""
    half_max = time_normalize / 2
    norm_offset = (time_offset - half_max) / half_max
    embeddings = []
    for b_time_offset in norm_offset:
        embeddings.append(
            _generate_position_encodings(
                b_time_offset[..., None], hidden_dim // 2, include_positions=False
            )
        )

    return torch.stack(embeddings)


class TimeDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        history_length: int = 16,
        class_decoder: bool = False,
    ) -> None:
        super().__init__()
        self.history_length = history_length
        self.hidden_dim = hidden_dim

        # Setup actual decoder
        self.value_tf = nn.Linear(input_dim, hidden_dim)
        self.time_attn = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)

        # Setup learned embedding to query time
        self.time_query = nn.Parameter(torch.empty(1, hidden_dim))
        with torch.no_grad():
            self.time_query.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

        # Setup decoding output of mhsa to a yaw estimate
        self.decode_yawrate = nn.Linear(hidden_dim, 11 if class_decoder else 1)

    def forward(self, im_feats: Tensor, time_encoding: Tensor) -> Tensor:
        bs = im_feats.shape[0]
        # time_encoding = self.time_encoding.expand([bs, -1, -1])
        time_query = self.time_query.expand([bs, -1, -1])
        values = self.value_tf(im_feats)

        yaw_embed, _ = self.time_attn(time_query, time_encoding, values)
        yaw_estimate = self.decode_yawrate(yaw_embed)
        return yaw_estimate


class TimeDecoder2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        history_length: int = 16,
        class_decoder: bool = False,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.history_length = history_length
        self.hidden_dim = hidden_dim
        self.sigmoid_output = sigmoid_output

        # Setup actual decoder
        self.feature_v = nn.Linear(input_dim, hidden_dim)
        self.feature_k = nn.Linear(input_dim, hidden_dim // 2)
        self.time_attn = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)

        # Setup learned embedding to query time
        self.time_query = nn.Parameter(torch.empty(1, hidden_dim))
        with torch.no_grad():
            self.time_query.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

        assert not (
            class_decoder and sigmoid_output
        ), f"Can't have class_decoder and sigmoid_output"

        # Setup decoding output of mhsa to a yaw estimate
        self.decode_yawrate = nn.Linear(hidden_dim, 11 if class_decoder else 1)

    def forward(self, im_feats: Tensor, time_encoding: Tensor) -> Tensor:
        bs = im_feats.shape[0]
        time_query = self.time_query.expand([bs, -1, -1])

        # Genrate Key/Values
        im_values = self.feature_v(im_feats)
        im_keys = self.feature_k(im_feats)
        comb_keys = torch.cat([time_encoding, im_keys], dim=-1)

        yaw_embed, _ = self.time_attn(time_query, comb_keys, im_values)
        yaw_estimate = self.decode_yawrate(yaw_embed)

        if self.sigmoid_output:
            yaw_estimate = torch.sigmoid(yaw_estimate) - 0.5

        return yaw_estimate


def show_timeenc(time_encoding):
    from matplotlib import pyplot as plt
    import matplotlib

    matplotlib.use("Agg")

    plt.figure(figsize=(20, 20))
    plt.imshow(time_encoding.detach())
    plt.savefig("timeenc.png")


def _forward_impl_patch(self, x: Tensor) -> Tensor:
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    return x


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False
            ),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)


class CustomV1(nn.Module):
    def __init__(self, out_channels: int = 256) -> None:
        super().__init__()
        self.conv_down = nn.Sequential(
            _DSConv(3, 8, stride=2),
            _DSConv(8, 24, stride=2),
            _DSConv(24, 32, stride=2),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(3)
        self.linear_out = nn.Linear(9 * 32, out_channels)
        self.output_dim = out_channels

    def forward(self, image: Tensor) -> Tensor:
        """"""
        feats: Tensor = self.conv_down(image)
        feats = self.avg_pool(feats)
        feats = self.linear_out(feats.flatten(1))
        return feats


class CustomV2(nn.Module):
    def __init__(self, out_channels: int = 128) -> None:
        super().__init__()
        self.conv_down = nn.Sequential(
            _DSConv(3, 8, stride=2),
            _DSConv(8, 24, stride=2),
            _DSConv(24, 32, stride=2),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d([2, 3])
        self.linear_out = nn.Linear(2 * 3 * 32, out_channels)
        self.output_dim = out_channels

    def forward(self, image: Tensor) -> Tensor:
        """"""
        feats: Tensor = self.conv_down(image)
        feats = self.avg_pool(feats)
        feats = self.linear_out(feats.flatten(1))
        return feats


class CustomV3(nn.Module):
    def __init__(self, out_channels: int = 128) -> None:
        super().__init__()
        self.conv_down = nn.Sequential(
            _DSConv(3, 8, stride=2),
            _DSConv(8, 24, stride=2),
            _DSConv(24, 16, stride=2),
        )
        self.avg_pool = nn.AvgPool2d([2, 2], [2, 2])
        # We assume latent feature of (h=3,w=5)
        self.linear_out = nn.Linear(3 * 5 * 16, out_channels)
        self.output_dim = out_channels

    def forward(self, image: Tensor) -> Tensor:
        """"""
        feats: Tensor = self.conv_down(image)
        feats = self.avg_pool(feats)
        feats = self.linear_out(feats.flatten(1))
        return feats


def get_encoder(name: str, pretrained: bool) -> MobileNetV3:
    if name == "small":
        return mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
    elif name == "tiny":
        return mobilenet_v3_tiny()
    elif name == "customv1":
        return CustomV1()
    elif name == "customv2":
        return CustomV2()
    elif name == "customv3":
        return CustomV3()
    else:
        raise NotImplementedError(name)


class SequenceModel(nn.Module):
    def __init__(
        self,
        max_history: float = 3,
        class_decoder: bool = False,
        pretrained: bool = True,
        attn_dim: int = 256,
        encoder: Literal["tiny", "small"] = "small",
    ) -> None:
        super().__init__()

        self.encoder = get_encoder(encoder, pretrained)

        # moneky patch to not classify
        if isinstance(self.encoder, MobileNetV3):
            self.encoder._forward_impl = _forward_impl_patch.__get__(
                self.encoder, MobileNetV3
            )
            self.encoder_dim = self.encoder.classifier[0].in_features
        else:
            self.encoder_dim = self.encoder.output_dim

        self.decoder = TimeDecoder(
            self.encoder_dim, hidden_dim=attn_dim, class_decoder=class_decoder
        )
        self.time_normalize = max_history

    def export_onnx(
        self, input_shape: Tuple[int, int], savedir: Path = Path.cwd()
    ) -> None:
        image = torch.empty((1, 3, *input_shape), dtype=torch.float32)
        torch.onnx.export(
            self.encoder,
            image,
            savedir / "encoder.onnx",
            opset_version=13,
            input_names=["image"],
            output_names=["features"],
        )

        tokens = torch.empty(
            (1, self.decoder.history_length, self.encoder_dim), dtype=torch.float32
        )
        timestamps = torch.empty(
            (1, self.decoder.history_length, self.decoder.hidden_dim),
            dtype=torch.float32,
        )
        torch.onnx.export(
            self.decoder,
            (tokens, timestamps),
            savedir / "decoder.onnx",
            opset_version=13,
            input_names=["features", "timestamps"],
            output_names=["yaw_rate"],
        )

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """Image stack [b,t,c,h,w]"""
        # Extract image features
        image_features = []
        for t_idx in range(inputs["image"].shape[1]):
            image_features.append(self.encoder(inputs["image"][:, t_idx]))
        image_features = torch.stack(image_features, dim=1)  # [b,t,c]

        # Generate time embeddings
        time_embeddings = _generate_embeddings(
            inputs["time"], self.time_normalize, self.decoder.hidden_dim
        )
        # show_timeenc(time_embeddings)

        # Predict yaw output
        yaw_pred = self.decoder(image_features, time_embeddings)[:, 0]  # [b,c]
        return yaw_pred


class SequenceModel2(SequenceModel):
    """Keys include both time and image features"""

    def __init__(
        self,
        *args,
        class_decoder: bool = False,
        attn_dim: int = 256,
        history_length: int = 16,
        sigmoid_output: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, class_decoder=class_decoder, **kwargs)
        self.decoder = TimeDecoder2(
            self.encoder_dim,
            hidden_dim=attn_dim,
            history_length=history_length,
            class_decoder=class_decoder,
            sigmoid_output=sigmoid_output,
        )

    def export_onnx(
        self, input_shape: Tuple[int, int], savedir: Path = Path.cwd()
    ) -> None:
        image = torch.empty((1, 3, *input_shape), dtype=torch.float32)
        torch.onnx.export(
            self.encoder,
            image,
            savedir / "encoder.onnx",
            opset_version=13,
            input_names=["image"],
            output_names=["features"],
        )

        tokens = torch.empty(
            (1, self.decoder.history_length, self.encoder_dim), dtype=torch.float32
        )
        timestamps = torch.empty(
            (1, self.decoder.history_length, self.decoder.hidden_dim // 2),
            dtype=torch.float32,
        )
        torch.onnx.export(
            self.decoder,
            (tokens, timestamps),
            savedir / "decoder.onnx",
            opset_version=13,
            input_names=["features", "timestamps"],
            output_names=["yaw_rate"],
        )

    def export_script(
        self, input_shape: Tuple[int, int], savedir: Path = Path.cwd()
    ) -> None:
        image = torch.empty((1, 3, *input_shape), dtype=torch.float32)
        script_enc = torch.jit.trace(self.encoder, image)
        torch.jit.save(script_enc, savedir / "encoder.jit")

        tokens = torch.empty(
            (1, self.decoder.history_length, self.encoder_dim), dtype=torch.float32
        )
        timestamps = torch.empty(
            (1, self.decoder.history_length, self.decoder.hidden_dim // 2),
            dtype=torch.float32,
        )
        script_dec = torch.jit.trace(self.decoder, (tokens, timestamps))
        torch.jit.save(script_dec, savedir / "decoder.jit")

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """Image stack [b,t,c,h,w]"""
        # Extract image features
        image_features = []
        for t_idx in range(inputs["image"].shape[1]):
            image_features.append(self.encoder(inputs["image"][:, t_idx]))
        image_features = torch.stack(image_features, dim=1)  # [b,t,c]

        # Generate time embeddings
        time_embeddings = _generate_embeddings(
            inputs["time"], self.time_normalize, self.decoder.hidden_dim // 2
        )

        # Predict yaw output
        yaw_pred = self.decoder(image_features, time_embeddings)[:, 0]  # [b,c]
        return yaw_pred


class SingleModel(nn.Module):
    def __init__(
        self, pretrained: bool = True, encoder: Literal["tiny", "small"] = "small"
    ) -> None:
        super().__init__()
        self.encoder = get_encoder(encoder, pretrained)

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
        "SequenceModel2": SequenceModel2,
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
    print(f"onnx time taken {time_taken:.3f}")


def export_sequence(model: SequenceModel) -> None:
    image = torch.empty((1, 3, 240, 320))
    tokens = torch.empty((1, model.decoder.history_length, model.encoder_dim))
    timestamps = torch.empty((1, model.decoder.history_length))

    torch.onnx.export(
        model.encoder, image, "encoder.onnx", opset_version=13, input_names=["image"]
    )

    torch.onnx.export(
        model.decoder,
        [tokens, timestamps],
        "decoder.onnx",
        opset_version=13,
        input_names=["features", "timestamps"],
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
    print(f"encoder time taken {time_taken:.3f}")

    # Decode yaw rate from features
    session = ort.InferenceSession("decoder.onnx")
    tokens = np.empty(session.get_inputs()[0].shape, dtype=np.float32)
    timestamps = np.empty(session.get_inputs()[1].shape, dtype=np.float32)
    print(f"Token shape: {tokens.shape}")
    session.run(None, {"features": tokens, "timestamps": timestamps})  # warm up

    start = time.perf_counter()
    for _ in range(niter):
        session.run(None, {"features": tokens, "timestamps": timestamps})
    time_taken = (time.perf_counter() - start) / niter
    print(f"decoder time taken {time_taken:.3f}")


def test_inference() -> None:
    niter = 5
    model = SequenceModel().eval()
    model.export_onnx([60, 80])
    # export_parts()
    profile_parts(niter)


if __name__ == "__main__":
    test_inference()
