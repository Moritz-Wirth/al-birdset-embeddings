from torch.utils.data import Dataset, default_collate
import datasets
from datasets import DatasetDict, ClassLabel, Sequence
import numpy as np
from typing import Union, Literal


class Reuters21578(Dataset):
    def __init__(self, split: Literal["train", "test"]):
        self.collate_fn = default_collate
        self.split = split
        self.ds = None

    @property
    def num_classes(self) -> int:
        assert self.ds is not None, "load dataset before accessing"
        return len(self.ds.features["topics"].feature.names)

    def load_dataset(self, **load_dataset_kwargs) -> None:
        # download dataset and set self.ds to dataset
        ds = datasets.load_dataset(**load_dataset_kwargs)
        names1 = {i for ii in ds["train"]["topics"] for i in ii}
        names2 = {i for ii in ds["test"]["topics"] for i in ii}
        names = names1.union(names2)
        ds = ds[self.split]
        ds = ds.select_columns(["title", "text", "topics"])
        ds = ds.cast_column("topics", Sequence(ClassLabel(names=sorted(names))))
        self.ds = ds

    def prepare_dataset(self) -> None:
        pass

    def __getitem__(self, idx: int) -> tuple:
        data = self.ds[idx]
        y = np.zeros(self.num_classes)
        y[data["topics"]] = 1
        return f"{data['title']}\n{data['text']}", y

    def __len__(self) -> int:
        assert self.ds is not None, "call \'load_dataset\' before accessing lengths"
        return len(self.ds["test_5s"]) if isinstance(self.ds, DatasetDict) else len(self.ds)
