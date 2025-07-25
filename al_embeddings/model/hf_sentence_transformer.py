import numpy as np
import os
from sentence_transformers import SentenceTransformer, models
from datasets.fingerprint import Hasher

from .base import ModelBase


class SentenceTransformerModel(ModelBase):
    def __init__(self, model_name: str, pooling_mode: str, cache_dir: str = None, normalize_embeddings: bool = False):
        cache_model = os.path.join(cache_dir, f"{model_name.replace('/', '-')}")
        transformer = models.Transformer(model_name,
                                         cache_dir=cache_model,
                                         model_args={"trust_remote_code": True},
                                         config_args={"trust_remote_code": True},
                                         tokenizer_args={"trust_remote_code": True})

        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode=pooling_mode)
        modules = [transformer, pooling]
        if normalize_embeddings:
            modules.append(models.Normalize())

        self.model = SentenceTransformer(modules=modules, trust_remote_code=True, cache_folder=cache_model)

    def __call__(self, x, *args, **kwargs) -> np.ndarray:
        self.model.cuda()
        return self.model.encode(x, show_progress_bar=False)

    @staticmethod
    def transforms(batch) -> tuple[np.ndarray, np.ndarray]:
        return batch

    def __getstate__(self):
        device = next(self.model.parameters()).device
        hasher = Hasher()
        model_on_cpu = self.model.cpu()
        for name, param in sorted(model_on_cpu.named_parameters()):
            hasher.update(param.cpu().detach().numpy())
        hasher.update(self.transforms)
        self.model.to(device)
        return hasher.hexdigest()


class BertBaseUncased(SentenceTransformerModel):
    def __init__(self, cache_dir: str = None, normalize_embeddings: bool = False):
        model_name: str = "google-bert/bert-base-uncased"
        pooling_mode = "cls"
        super().__init__(model_name=model_name,
                         pooling_mode=pooling_mode,
                         cache_dir=cache_dir,
                         normalize_embeddings=normalize_embeddings)