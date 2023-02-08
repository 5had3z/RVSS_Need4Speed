"""
Testing visually what the waypoint deteciton looks like.

Detect white boundary of page, reproject/warp page as tile, use mobilenetv3 to classify
which type of page it is (have all orentations of all pages as classes).

How good can white deliniation can be?
"""
import argparse
from pathlib import Path
from typing import Callable, Dict

import cv2
import numpy as np


def harris_corner(img: np.ndarray) -> np.ndarray:
    """Use harris corner detector for road edges"""
    cv2.cornerHarris(img, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    sorted = np.argsort(dst)
    k = 10
    topk = np.array([np.unravel_index(point, img.shape) for point in sorted[:k]])
    return topk


ALGORITHMS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {"harris": harris_corner}


def script_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, help="Folder to detect waypoints in")
    parser.add_argument("--image", type=Path, help="Run detection on individual image")
    parser.add_argument(
        "--output", type=Path, help="Where to write overlayed predictions to"
    )
    parser.add_argument(
        "--algo", type=str, help=f"Select algorithm from {ALGORITHMS.keys()}"
    )
    return parser.parse_args()


def run_deteciton(
    img: np.ndarray, algorihtm: Callable[[np.ndarray], np.ndarray], writeFile: Path
) -> None:
    """Add waypoint visualisation to image"""
    waypoints = algorihtm(img)

    for waypoint in waypoints:
        img = cv2.drawMarker(
            img,
            waypoint,
            (0, 0, 255),
            cv2.MARKER_TILTED_CROSS,
            markerSize=6,
            thickness=2,
        )

    cv2.imwrite(str(writeFile), img)

def detect_image(img: Path, outFolder: Path, algorithm: Callable[[np.ndarray], np.ndarray]) -> None:
    """"""
    
def detect_folder(root: Path, outFolder: Path, algorithm: Callable[[np.ndarray], np.ndarray]) -> None;
    """"""
    
    

def main() -> None:
    """main entrypoint to the test"""
    args = script_args()
    algo = ALGORITHMS[args.algo]
    if args.folder:
        detect_folder(args.folder, args.output, algo)
    elif args.image:
        detect_image(args.image, args.output, algo)
    else:
        raise RuntimeError()


if __name__ == "__main__":
    main()
