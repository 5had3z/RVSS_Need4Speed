from pathlib import Path
from typing import Optional, Tuple
import logging

import cv2
import numpy as np
import onnxruntime as ort


def _generate_position_encodings(
    p: np.ndarray,
    num_frequency_bands: int,
    max_frequencies: Optional[Tuple[int, ...]] = None,
    include_positions: bool = True,
) -> np.ndarray:
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
        np.linspace(1.0, max_freq / 2.0, num_frequency_bands)
        for max_freq in max_frequencies
    ]
    frequency_grids = []

    for i, frequencies_i in enumerate(frequencies):
        frequency_grids.append(p[..., i : i + 1] * frequencies_i[None, ...])

    if include_positions:
        encodings.append(p)

    encodings.extend(
        [np.sin(np.pi * frequency_grid) for frequency_grid in frequency_grids]
    )
    encodings.extend(
        [np.cos(np.pi * frequency_grid) for frequency_grid in frequency_grids]
    )

    return np.concatenate(encodings, axis=-1).astype(np.float32)


def _generate_embeddings(
    time_offset: np.ndarray, time_normalize: float, hidden_dim: int
) -> np.ndarray:
    """Generate feature embedding for time offsets"""
    half_max = time_normalize / 2
    norm_offset = (time_offset - half_max) / half_max

    return _generate_position_encodings(
        norm_offset[..., None],
        hidden_dim // 2,
        include_positions=False,
    )[None]


class TemporalModel:
    def __init__(self, weights_root: Path) -> None:
        self.logger = logging.getLogger("model")
        self.current_idx = 0
        self.buffer_filled = False
        self.time_norm = 1.5
        self.logger.info(f"Using Time Normalization {self.time_norm}")

        # Setup Encoder
        self.logger.info("Initializing Encoder...")
        self.encoder_session = ort.InferenceSession(str(weights_root / "encoder.onnx"))

        # Setup decoder
        self.logger.info("Initializing Decoder...")
        self.decoder_session = ort.InferenceSession(str(weights_root / "decoder.onnx"))
        self.feature_buffer = np.empty(
            self.decoder_session.get_inputs()[0].shape, dtype=np.float32
        )
        self.time_ch = self.decoder_session.get_inputs()[1].shape[-1]

        self.buffer_size = self.feature_buffer.shape[1]
        self.timestamp_buffer = self.buffer_size * [None]
        self.logger.info(f"Temporal Buffer Size: {self.buffer_size}")

    def generate_timestamp_offsets(self) -> np.ndarray:
        end_time = self.timestamp_buffer[self.current_idx]
        return np.array([end_time - t for t in self.timestamp_buffer], dtype=np.float32)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """preprocess image for inference"""
        assert (
            len(image.shape) == 3 and image.shape[-1] == 3
        ), f"Expected HWC, got {image.shape}"

        # Transform to float image
        image = image.astype(np.float32) / 255

        # Resize
        infer_shape = self.encoder_session.get_inputs()[0].shape[-2:][::-1]
        image = cv2.resize(image, infer_shape, interpolation=cv2.INTER_LINEAR)

        # Normalize
        for idx, (mu, sgma) in enumerate(
            zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ):
            image[:, :, idx] = (image[:, :, idx] - mu) / sgma

        image = np.moveaxis(image, -1, 0)[None]
        return image

    def run_inference(self, image, timestamp) -> Optional[float]:
        image_tf = self.preprocess_image(image)
        self.feature_buffer[:, self.current_idx] = self.encoder_session.run(
            None, {"image": image_tf}
        )[0]
        self.timestamp_buffer[self.current_idx] = timestamp

        if self.buffer_filled:
            time_deltas = self.generate_timestamp_offsets()
            time_embeddings = _generate_embeddings(
                time_deltas, self.time_norm, self.time_ch
            )
            yaw_ctrl = self.decoder_session.run(
                None, {"features": self.feature_buffer, "timestamps": time_embeddings}
            )
            yaw_ctrl = (
                (np.argmax(yaw_ctrl) - 5).item() / 10
                if yaw_ctrl.shape[-1] > 1
                else yaw_ctrl.item()
            )
        else:
            yaw_ctrl = None

        self.current_idx = (self.current_idx + 1) % self.buffer_size

        if not self.buffer_filled and self.current_idx == 0:
            self.buffer_filled = True
            self.logger.info("Buffer Full! Performing inference next step")

        return yaw_ctrl
