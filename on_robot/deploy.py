#!/usr/bin/env python3
import logging
from pathlib import Path

import numpy as np

import penguin_pi as ppi
from temporal_model import TemporalModel


class Robot:
    Kd = 30  # base wheel speeds, increase to go faster, decrease to go slower
    Ka = 30  # how fast to turn when given an angle

    def __init__(self) -> None:
        self.logger = logging.getLogger("robot")

        self.logger.info("Initializing Model...")
        self.model = TemporalModel(Path.cwd())

        self.logger.info("Initializing Camera...")
        self.camera = ppi.VideoStreamWidget("http://localhost:8080/camera/get")

    def _steering_command(self, angle: float) -> None:
        angle = np.clip(angle, -0.5, 0.5)
        left = int(self.Kd + self.Ka * angle)
        right = int(self.Kd - self.Ka * angle)

        ppi.set_velocity(left, right)

    def spin(self) -> None:
        """"""
        while True:
            image, ts = self.camera.get_frame()
            if any(x is None for x in [image, ts]):
                self.logger.info(f"tried to get frame")
                continue

            self.logger.info(f"got frame: {ts}")
            yaw = self.model.run_inference(image, ts)

            if yaw is None:  # Skip first few images
                continue

            self.logger.info(f"steering command: {yaw}")
            self._steering_command(yaw)

    def stop(self) -> None:
        """"""
        ppi.set_velocity(0, 0)


def main() -> None:
    ppi.set_velocity(0, 0)  # ensure stopped
    robot = Robot()
    try:
        robot.spin()
    except KeyboardInterrupt:
        pass
    finally:
        robot.stop()


if __name__ == "__main__":
    main()
