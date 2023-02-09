from pathlib import Path
from typing import Optional
import logging

import cv2
import numpy as np
import onnxruntime as ort


class TemporalModel:
    def __init__(self, weights_root: Path) -> None:
        self.logger = logging.getLogger("model")
        self.current_idx = 0
        self.buffer_filled = False

        # Setup decoder
        self.logger.info("Initializing Decoder")
        self.decoder_session = ort.InferenceSession(weights_root / "decoder.onnx")
        self.feature_buffer = np.empty(
            self.decoder_session.get_inputs()[0].shape, dtype=np.float32
        )
        self.timestamp_buffer = []

        # Setup Encoder
        self.logger.info("Initializing Encoder")
        self.encoder_session = ort.InferenceSession(weights_root / "encoder.onnx")

        self.buffer_size = self.feature_buffer.shape[1]
        self.logger.info(f"Temporal Buffer Size: {self.buffer_size}")

    def generate_timestamp_offsets(self) -> np.ndarray:
        end_time = self.timestamp_buffer[self.current_idx]
        return np.array([end_time - t for t in self.timestamp_buffer], dtype=np.float32)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """preprocess image for inference"""
        # Transform to float image
        image = image.astype(np.float32) / 255

        # Resize
        infer_shape = self.encoder_session.get_inputs()[0].shape[-2:]
        image = cv2.resize(image, infer_shape, interpolation=cv2.INTER_LINEAR)

        # Normalize
        for idx, (mu, sgma) in enumerate(
            zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ):
            image[:, idx] = (image[:, idx] - mu) / sgma

        return image

    def run_inference(self, image, timestamp) -> Optional[float]:
        image_tf = self.preprocess_image(image)
        self.feature_buffer[self.current_idx] = self.encoder_session.run(
            None, {"image": image_tf}
        )
        self.timestamp_buffer[self.current_idx] = timestamp

        if self.buffer_filled:
            time_deltas = self.generate_timestamp_offsets()
            yaw_ctrl = self.decoder_session.run(
                None, {"features": self.feature_buffer, "timestamps": time_deltas}
            )
        else:
            yaw_ctrl = None

        self.current_idx = (self.current_idx + 1) % self.buffer_size

        if not self.buffer_filled and self.current_idx == 0:
            self.buffer_filled = True
            self.logger.info("Buffer Full! Performing inference next step")

        return yaw_ctrl
