from torch.utils.data import Dataset, default_collate
import datasets
from datasets import Audio, DatasetDict
import numpy as np
from al_embeddings.utils.data_collators import DataCollatorWithPadding


class Birdset(Dataset):
    def __init__(self):
        self.collate_fn = DataCollatorWithPadding()
        self.ds = None

    @property
    def num_classes(self) -> int:
        assert self.ds is not None, "call \'load_dataset\' before accessing num_classes"
        ds = self.ds["test_5s"] if isinstance(self.ds, DatasetDict) else self.ds
        return len(ds.features["ebird_code_multilabel"].feature.names)

    def load_dataset(self, **load_dataset_kwargs) -> None:
        # download dataset and set self.ds to dataset
        self.ds = datasets.load_dataset(**load_dataset_kwargs)

    def prepare_dataset(self) -> None:
        # dataset specific preparation
        ds = self.ds["test_5s"] if isinstance(self.ds, DatasetDict) else self.ds
        ds = ds.cast_column("audio", Audio(decode=True, mono=True, sampling_rate=32_000))
        ds = ds.select_columns(["audio", "ebird_code_multilabel"])
        self.ds = ds

    def __getitem__(self, idx: int) -> tuple:
        data = self.ds[idx]
        y = np.zeros(self.num_classes)
        y[data["ebird_code_multilabel"]] = 1
        return data["audio"]["array"], y

    def __len__(self) -> int:
        assert self.ds is not None, "call \'load_dataset\' before accessing lengths"
        return len(self.ds["test_5s"]) if isinstance(self.ds, DatasetDict) else len(self.ds)
