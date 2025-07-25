from perch_hoplite.zoo.model_configs import load_model_by_name
from .base import ModelBase
from datasets.fingerprint import Hasher


class PerchEmbedModel(ModelBase):
    def __init__(self, model_name: str = "perch_8"):

        self.model = load_model_by_name(model_name)
        self.model_name = model_name

    def __getstate__(self):
        hasher = Hasher()
        for variable in self.model.model._variables:
            hasher.hash(variable)

        hasher.hash(self.model_name)
        return hasher.hexdigest()

    def __call__(self, audio_array, *args, **kwargs):
        return self.model.embed(audio_array).embeddings.squeeze()
