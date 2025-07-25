from torch.utils.data import Dataset, default_collate
import requests
import os
import numpy as np
from typing import Literal
from tqdm import tqdm
import zipfile


class AAPD(Dataset):
    def __init__(self, split: Literal["train", "test"]):
        self.X = None
        self.y = None
        self.split = split
        self.collate_fn = default_collate
        self._url = "https://www.kaggle.com/api/v1/datasets/download/xiaojuanwang9/aapd-dataset"
        self.label_map = None

    @property
    def num_classes(self) -> int:
        return len(self.label_map) # 54

    def load_dataset(self, cache_dir: str = None) -> None:
        # download dataset and set self.ds to dataset
        cache_dir = cache_dir or f"./data/{self.__class__.__name__}"
        os.makedirs(cache_dir, exist_ok=True)
        zip_path = os.path.join(cache_dir, "aapd.zip")

        self._download(zip_path)
        self._extract(zip_path)

        data_path = os.path.join(cache_dir, f"{self.split}.txt")

        with open(data_path, "r", encoding="utf-8") as file:
            data = file.readlines()

        self.X = data[::2]

        label_text = data[1::2]
        label_list = [[l.lower().replace("\n", "") for l in label.split(" ")] for label in label_text]
        label_map = {i for ii in label_list for i in ii}
        self.label_map = dict(map(reversed, enumerate(sorted(label_map))))
        self.y = list(map(lambda l: [self.label_map[x] for x in l], label_list))

    def _download(self, zip_path: str):
        # check of download already has been done
        if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
            return

        response = requests.get(self._url, stream=True)
        response.raise_for_status()
        try:
            with open(zip_path, "wb") as file:
                for chunk in tqdm(response.iter_content(chunk_size=8192)):
                    if chunk:
                        file.write(chunk)
        except requests.exceptions.RequestException as e:
            if os.path.exists(zip_path):
                os.remove(zip_path)  # Clean up partial download
            raise
        except Exception as e:
            if os.path.exists(zip_path):
                os.remove(zip_path)  # Clean up partial download
            raise

    def _extract(self, zip_path: str):
        base = os.path.dirname(zip_path)
        dataset_path = [os.path.join(base, "train.txt"), os.path.join(base, "test.txt")]
        if all(os.path.exists(path) and os.path.getsize(path) > 0 for path in dataset_path):
            return

        try:
            with zipfile.ZipFile(zip_path, "r") as zip:
                zip.extractall(base)
        except zipfile.BadZipFile:
            print(f"Error: '{zip_path}' is not a valid zip file. It might be corrupted or incomplete.")
            raise
        except Exception as e:
            print(f"An error occurred during extraction: {e}")
            raise

    def prepare_dataset(self) -> None:
        # dataset specific preparation
        pass

    def __getitem__(self, idx: int) -> tuple:
        X = self.X[idx]
        y = np.zeros(self.num_classes)
        y[self.y[idx]] = 1
        return X, y

    def __len__(self) -> int:
        assert self.X is not None, "call \'load_dataset\' before accessing lengths"
        return len(self.X) - 1

    # @staticmethod
    # def collator_fn(batch):
    #     return default_collate
