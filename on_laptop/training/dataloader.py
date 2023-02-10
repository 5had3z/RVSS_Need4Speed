from pathlib import Path
import os
from typing import List, Tuple, Literal
from dataclasses import dataclass
from datetime import datetime, timedelta

from PIL import Image
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
        return cls(data[0], float(data[1]), datetime.fromisoformat(data[2]))

    def __str__(self) -> str:
        return f"{self.filename},{self.yaw},{self.time}"


def read_dataset(ann_file: Path) -> List[YawRateSample]:
    """"""
    dataset = []
    with open(ann_file, "r") as f:
        while line := f.readline():
            dataset.append(YawRateSample.from_line(line))
    return dataset


def write_dataset(dataset: List[YawRateSample], ann_file: Path) -> None:
    """"""
    print(f"writing {ann_file}")
    with open(ann_file, "w") as f:
        for sample in dataset:
            f.write(f"{sample}\n")


class CropTop:
    def __init__(self, first_row: int) -> None:
        self.first_row = first_row

    def __call__(self, img: Tensor) -> Tensor:
        """Assume CHW"""
        return img[:, self.first_row :, :]


class YawDataset(Dataset):
    base_shape = [240, 320]
    ch_last = False

    def __init__(
        self,
        split: Literal["train", "val"],
        downsample: int = 1,
        classes: bool = False,
        suffix: str = None,
        crop_top: bool = False,
    ) -> None:
        super().__init__()
        assert split in {"train", "val"}, f"Incorrect split: {split}"

        self.root = Path(os.environ.get("DATA_ROOT", "/data"))
        filename = (
            f"annotation_{split}.txt"
            if suffix is None
            else f"annotation_{split}_{suffix}.txt"
        )
        self.dataset: List[YawRateSample] = read_dataset(self.root / filename)
        self.dataset.sort(key=lambda x: x.time)

        for sample in self.dataset:
            assert (self.root / sample.filename).exists()

        transforms = [transform.ToTensor()]

        if downsample > 1:
            transforms.append(
                transform.Resize([s // downsample for s in YawDataset.base_shape])
            )

        if crop_top:
            transforms.append(CropTop(15))

        transforms.append(
            transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
        if split == "train":
            transforms.append(transform.ColorJitter(0.2, 0.1, 0.1, 0.1))

        self.transform = transform.Compose(transforms)
        self.classes = classes

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _yaw_2_class(yaw: Tensor) -> Tensor:
        """Convert yaw to classes"""
        yaw = torch.round(yaw * 10) + 5
        return yaw.to(torch.long)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        sample = self.dataset[index]
        im = Image.open(str(self.root / sample.filename))
        im = self.transform(im)

        label = tensor([sample.yaw])
        if self.classes:
            label = self._yaw_2_class(label)

        if self.ch_last:
            im = im.to(memory_format=torch.channels_last)

        return {"image": im, "yaw": label}


class YawSequenceDataset(YawDataset):
    def __init__(
        self, *args, max_period_ms: float = 150.0, seq_length: int = 16, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.seq_length = seq_length

        self.dataset: List[List[YawRateSample]] = self.create_sequence_dataset(
            self.dataset, timedelta(milliseconds=max_period_ms)
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
            if len(new_seq) == 0:
                new_seq.append(sample)
                continue

            time_diff = sample.time - new_seq[-1].time
            if time_diff < max_period and time_diff > timedelta(0):
                new_seq.append(sample)
            else:  # no neighbour, reset
                new_seq = []

        return new_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        ims = []
        for sample in self.dataset[index]:
            im = Image.open(str(self.root / sample.filename))
            ims.append(self.transform(im))
            if self.ch_last:
                ims[-1] = ims[-1].to(memory_format=torch.channels_last)

        end_time = self.dataset[index][-1].time.timestamp()
        t_delta = tensor([end_time - s.time.timestamp() for s in self.dataset[index]])
        assert torch.all(t_delta >= 0), "negative delta???"

        ims = torch.stack(ims, dim=0)
        label = tensor(self.dataset[index][-1].yaw)

        if self.classes:
            label = self._yaw_2_class(label)

        return {"image": ims, "time": t_delta, "yaw": label}


class YawSequenceDataset2(YawDataset):
    def __init__(
        self,
        *args,
        max_period_ms: float = 150.0,
        seq_length: int = 16,
        stride: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.stride = stride
        self.seq_length = seq_length
        self.max_period = timedelta(milliseconds=max_period_ms)

    def __len__(self) -> int:
        return (len(self.dataset) - self.seq_length) // self.stride

    def get_next_valid_sequence(self, index) -> int:
        has_wrapped = False
        cond = lambda nxt, prv: nxt - prv < self.max_period and nxt - prv > timedelta(0)

        while True:
            if index == len(self):
                if has_wrapped:
                    raise RecursionError("Iterated over whole dataset")
                index = 0
                has_wrapped = True

            end_idx = index + self.seq_length
            if all(
                cond(x.time, y.time)
                for x, y in zip(
                    self.dataset[index + 1 : end_idx], self.dataset[index : end_idx - 1]
                )
            ):
                return index

            index += 1

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        s_idx = self.get_next_valid_sequence(index * self.stride)
        e_idx = s_idx + self.seq_length
        ims = []
        for idx in range(s_idx, e_idx):
            sample = self.dataset[idx]
            im = Image.open(str(self.root / sample.filename))
            ims.append(self.transform(im))
            if self.ch_last:
                ims[-1] = ims[-1].to(memory_format=torch.channels_last)

        end_time = self.dataset[e_idx - 1].time.timestamp()
        t_delta = tensor(
            [end_time - s.time.timestamp() for s in self.dataset[s_idx:e_idx]]
        )
        assert torch.all(t_delta >= 0), "negative delta???"

        ims = torch.stack(ims, dim=0)
        label = tensor(self.dataset[e_idx].yaw)

        if self.classes:
            label = self._yaw_2_class(label)

        return {"image": ims, "time": t_delta, "yaw": label}


def get_dataloader(config) -> Tuple[Dataset, Dataset]:
    """Return train, val"""
    dataset_types = {
        "YawDataset": YawDataset,
        "YawSequenceDataset": YawSequenceDataset,
        "YawSequenceDataset2": YawSequenceDataset2,
    }

    dataset_cfg = config["dataloader"]["dataset"]
    loader_kwargs = dict(
        drop_last=True,
        batch_size=config["dataloader"]["batch_size"],
        num_workers=config["dataloader"]["workers"],
    )
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


def accumulate_subsets(root: Path) -> List[YawRateSample]:
    all_data: List[YawRateSample] = []
    for ann_file in root.glob("**/annotation.txt"):
        print(f"reading {ann_file}")
        dataset = read_dataset(ann_file)
        new_prefix = ann_file.relative_to(root).parent
        for sample in dataset:
            sample.filename = str(new_prefix / sample.filename)

        all_data.extend(dataset)

    assert len(all_data) > 0, "No data found"
    print(f"Collected {len(all_data)} samples")
    return all_data


def split_data() -> None:
    """"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--root", type=Path, help="Dataset annotation file to split"
    )
    parser.add_argument("-s", "--split", type=float, help="Split Ratio", default=0.8)
    args = parser.parse_args()

    root: Path = args.root
    ratio: float = args.split
    assert 0 < ratio < 1, f"Train ratio needs to be 0<{ratio}<1"

    data = accumulate_subsets(root)

    # use interleaving method to split
    chunk_size = 400
    train_data = []
    val_data = []
    n_train = int(chunk_size * ratio)
    for s_idx in range(0, len(data), chunk_size):
        val_idx = s_idx + n_train
        train_data.extend(data[s_idx:val_idx])
        val_data.extend(data[val_idx : s_idx + chunk_size])

    leftover_idx = len(data) % chunk_size
    train_data.extend(data[-leftover_idx:])

    split_file = root / f"annotation_train_v4.txt"
    write_dataset(train_data, split_file)

    split_file = root / f"annotation_val_v4.txt"
    write_dataset(val_data, split_file)

    print(f"finished splitting {len(data)} samples")


def analyse_data() -> None:
    """"""
    import numpy as np
    import os
    from tqdm.auto import tqdm

    os.environ[
        "DATA_ROOT"
    ] = "/home/bpfer/cloned-repos/RVSS_Need4Speed/on_laptop/training/data/first_run"

    cfg = {
        "dataloader": {
            "dataset": {"type": "YawSequenceDataset2", "args": {"suffix": "v4"}},
            "batch_size": 16,
            "workers": 8,
        }
    }
    train, val = get_dataloader(cfg)

    for split in [train, val]:
        angles = []
        for data in tqdm(split, ncols=50):
            angles.append(torch.round(data["yaw"] * 10).numpy())
        angles = np.concatenate(angles).astype(np.int64)
        count = np.bincount(angles + 5)
        print(count)
        print(1 - count / count.sum())


if __name__ == "__main__":
    split_data()
    # analyse_data()
