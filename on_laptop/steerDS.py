from pathlib import Path
from typing import List

import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class SteerDataSet(Dataset):
    def __init__(self, root_folder: Path, img_ext: str = "jpg", transform=None):
        self.img_ext = img_ext
        self.root_folder = root_folder
        self.transform = transforms.ToTensor() if transform is None else transform
        self.filenames: List[Path] = list(Path.glob(f"{root_folder}/*.{self.img_ext}"))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        f: Path = self.filenames[idx]
        img = cv2.imread(str(f))
        img = self.transform(img)
        steering = np.float32(f.stem[6:])
        return {"image": img, "steering": steering}


def test():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    ds = SteerDataSet("~/RVSS_Need4Speed/on_laptop/data", ".jpg", transform)

    print(f"The dataset contains {len(ds)} images ")

    ds_dataloader = DataLoader(ds, batch_size=1, shuffle=True)
    for s in ds_dataloader:
        im = s["image"]
        y = s["steering"]
        print(f"{im.shape} {y}")
        break


if __name__ == "__main__":
    test()
