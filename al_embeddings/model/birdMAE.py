import numpy as np

from .base import ModelBase
from datasets.fingerprint import Hasher
from transformers import AutoModel, AutoFeatureExtractor

class BirdMAEEmbedModel(ModelBase):
    def __init__(self, model_name: str = "DBD-research-group/Bird-MAE-Huge", cache_dir: str = None):

        self.model = AutoModel.from_pretrained(model_name,
                                               cache_dir=cache_dir,
                                               trust_remote_code=True)
        self.model_name = model_name
        self.fe = AutoFeatureExtractor.from_pretrained(model_name,
                                                       cache_dir=cache_dir,
                                                       trust_remote_code=True)

    def __getstate__(self):
        hasher = Hasher()
        for param in self.model.parameters():
            hasher.hash(param)

        hasher.hash(self.model_name)
        return hasher.hexdigest()

    def transforms(self, batch) -> tuple[np.ndarray, np.ndarray]:
        return self.fe(batch[0]).unsqueeze(1), batch[1]


    def __call__(self, audio_array, *args, **kwargs):
        return self.model(audio_array)["last_hidden_state"].squeeze().detach().numpy()
