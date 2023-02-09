from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple
import yaml

from trainer import get_model


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-x", "--experiment", type=Path, help="Path to experiment folder"
    )
    parser.add_argument("--input_shape", type=int, nargs=2, help="[h,w] of image")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    exp_path: Path = args.experiment
    img_shape: Tuple[int, int] = args.input_shape
    print(f"Exporting with image input of {img_shape}")
    with open(exp_path / "config.yml", "r") as f:
        config = yaml.safe_load(f)
    model = get_model(config)
    model.export_onnx(img_shape, exp_path)


if __name__ == "__main__":
    main()
