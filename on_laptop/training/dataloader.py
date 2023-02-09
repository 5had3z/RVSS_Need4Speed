from pathlib import Path
import os
from typing import List, Tuple, Literal
from dataclasses import dataclass
from datetime import datetime

from PIL.Image import Image
import torchvision.transforms as transform
import torch
from torch import Tensor, tensor
from torch.utils.data import Dataset, DataLoader


@dataclass
class YawRateSample:
    filename: str
    yaw: float
    time: datetime

    @classmethod
    def from_line(cls, line: str):
        data = line.strip().split(",")
        return cls(data[0], data[1], datetime.fromisoformat(data[2]))

    def __str__(self) -> str:
        return f"{self.filename},{self.yaw},{self.time}"


class YawDataset(Dataset):
    def __init__(self, split: Literal["train", "val"]) -> None:
        super().__init__()
        assert split in {"train", "val"}, f"Incorrect split: {split}"

        self.root = Path(os.environ.get("DATA_ROOT", "/data"))
        self.dataset: List[YawRateSample] = []
        with open(self.root / f"annotations_{split}.txt", "r") as f:
            while line := f.readline():
                self.dataset.append(YawRateSample.from_line(line))
        self.dataset.sort(key=lambda x: x.time)

        for sample in self.dataset:
            assert (self.root / sample.filename).exists()

        transforms = [
            transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        if split == "train":
            transforms.append(transform.ColorJitter(0.5, 0.2, 0.2, 0.2))

        transforms.append(transform.ToTensor())

        self.transform = transform.Compose(*transforms)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        sample = self.dataset[index]
        im = Image(str(self.root / sample.filename))
        im = self.transform(im)
        label = tensor(sample.yaw)
        return im, label


class YawSequenceDataset(YawDataset):
    def __init__(
        self, *args, frequency: float = 10.0, seq_length: int = 16, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.seq_length = seq_length

        self.dataset: List[List[YawRateSample]] = self.create_sequence_dataset(
            self.dataset, 1 / frequency
        )

    def create_sequence_dataset(
        self, old_dataset: List[YawRateSample], max_period: float
    ) -> List[List[YawRateSample]]:
        new_dataset: List[List[YawRateSample]] = []

        new_seq: List[YawRateSample] = []
        for sample in old_dataset:
            if len(new_seq) == self.seq_length:
                new_dataset.append(new_seq)
                new_seq = []
            if new_seq[-1].time - sample.time < max_period:
                new_seq.append(sample)
            else:  # no neighbour, reset
                new_seq = []

        return new_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        ims = []
        for sample in self.dataset[index]:
            im = Image(str(self.root / sample.filename))
            ims.append(self.transform(im))
        ims = torch.stack(ims, dim=0)
        label = tensor(self.dataset[index][-1].yaw)
        return ims, label


def get_dataloader(config) -> Tuple[Dataset, Dataset]:
    """Return train, val"""
    dataset_types = {
        "YawDataset": YawDataset,
        "YawSequenceDataset": YawSequenceDataset,
    }

    dataset_cfg = config["dataloader"]["dataset"]
    loader_kwargs = dict(batch_size=config["dataloader"]["batch_size"], num_workers=2)
    dataset_type: YawDataset = dataset_types[dataset_cfg["type"]]
    train_loader = DataLoader(
        dataset_type(split="train", **dataset_cfg["args"]),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        dataset_type(split="val", **dataset_cfg["args"]), **loader_kwargs
    )

    return train_loader, val_loader


def split_data() -> None:
    """"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--ann", type=Path, help="Dataset annotation file to split"
    )
    parser.add_argument("-r", "--ratio", type=float, help="Fraction of train split")
    args = parser.parse_args()

    ann_file: Path = args.ann
    ratio: float = args.ratio
    assert 0 < ratio < 1, f"Train ratio needs to be 0<{ratio}<1"

    with open(ann_file, "r") as f:
        data = f.readlines()

    split_file = ann_file.parent / f"{ann_file.stem}_train.txt"
    n_train = int(len(data) * ratio)
    with open(split_file, "w") as f:
        f.writelines(data[:n_train])

    split_file = ann_file.parent / f"{ann_file.stem}_val.txt"
    with open(split_file, "w") as f:
        f.writelines(data[n_train:])

    print(f"finished splitting {len(data)} samples")


if __name__ == "__main__":
    split_data()
