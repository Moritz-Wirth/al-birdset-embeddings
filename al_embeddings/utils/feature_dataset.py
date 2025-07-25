import numpy as np
from al_embeddings.model.base import ModelBase
from datasets.fingerprint import Hasher
import os
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from al_embeddings.utils import Compose


class FeatureDataset:
    def __init__(self,
                 dataset: Dataset,
                 model: ModelBase,
                 cache_dir: str | None = None,
                 dataloader_kwargs: dict = None):
        self.dataset = dataset
        self.model = model
        self.dataloader_kwargs = dataloader_kwargs or {}
        self.X = None
        self.y = None
        self.embed_file = None
        self.label_file = None

        if cache_dir is not None:
            self.cache_dir = os.path.join(cache_dir,
                                          dataset.__class__.__name__,
                                          model.__class__.__name__)

            self.embed_file = os.path.join(self.cache_dir, f"{self._fingerprint}_embeds.npy")
            self.label_file = os.path.join(self.cache_dir, f"{self._fingerprint}_labels.npy")

            if os.path.exists(self.embed_file) and os.path.exists(self.label_file):
                print(f"Loading embeddings from {self.embed_file}.")
                self.X = np.load(self.embed_file)
                print(f"Loading labels from {self.label_file}.")
                self.y = np.load(self.label_file)
            else:
                print(f"Did not find embeddings and labels cached.")

    @property
    def _fingerprint(self):
        hasher = Hasher()
        hasher.update(self.dataset)
        hasher.update(self.model)
        return hasher.hexdigest()

    def build(self) -> tuple[np.ndarray, np.ndarray]:
        if self.X is None and self.y is None:
            self.compute_embeddings()
        return self.X, self.y

    def __len__(self) -> int:
        return len(self.dataset)

    @torch.no_grad()
    def compute_embeddings(self):
        collate_fn = Compose([self.dataset.collate_fn,  self.model.transforms])
        dataloader = DataLoader(self.dataset, **self.dataloader_kwargs, collate_fn=collate_fn)

        embeddings = []
        labels = []

        for X, y in tqdm(dataloader, desc="Creating Embeddings"):
            X = self.model(X)
            embeddings.append(X)
            labels.append(y)

        self.X = np.concatenate(embeddings)
        self.y = np.concatenate(labels)

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Saving embeddings to {self.embed_file}")
            np.save(self.embed_file, self.X)
            print(f"Saving labels to {self.embed_file}")
            np.save(self.label_file, self.y)

    # TODO
    def split_data(self, test_size: int, stratified: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass
