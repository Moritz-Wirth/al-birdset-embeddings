from abc import ABC, abstractmethod

import numpy as np


class ModelBase(ABC):
    """
    In case this class is not hash able implement the __getstate__ method.
    This method should return the current state of the model as a hashable (hash, tuple, dict, ...)
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, x, *args, **kwargs) -> np.ndarray:
        """Calculates embeddings of x"""
        pass

    @staticmethod
    def transforms(batch) -> tuple[np.ndarray, np.ndarray]:
        """model specific transforms"""
        return batch