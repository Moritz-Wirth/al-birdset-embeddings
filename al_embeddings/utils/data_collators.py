import numpy as np
import torch
from typing import Literal, Any


class DataCollatorWithPadding:
    def __init__(self, max_length: int = None, return_tensor: Literal["pt", "np"] = "np"):
        self.max_length = max_length
        self.return_tensor = return_tensor

    def __call__(self, batch: list[tuple[np.ndarray | torch.Tensor, Any]]) -> tuple[torch.Tensor | np.ndarray, Any]:
        """
        Collates a list of samples into a batch with padding and truncation.

        Args:
            batch (`list`): A list of samples, where each sample is expected
                           to be a tuple (audio_array, label).

        Returns:
            A tuple containing the batched audio data and labels,
            either as PyTorch tensors or NumPy arrays.
        """
        audio_data, labels = zip(*batch)

        processed_audio = []
        lengths = []
        for audio in audio_data:
            if not isinstance(audio, torch.Tensor):
                audio = torch.tensor(audio)
            if self.max_length is not None and audio.shape[0] > self.max_length:
                audio = audio[:self.max_length]
            lengths.append(audio.shape[0])
            processed_audio.append(audio)

        max_length = max(lengths)
        batched_audio = []
        for audio in processed_audio:
            batched_audio.append(torch.nn.functional.pad(audio, (0, max_length - audio.shape[0])))

        batched_audio = torch.stack(batched_audio)
        batched_labels = torch.tensor(labels)

        if self.return_tensor == "np":
            if not isinstance(batched_audio, list):
                 batched_audio = batched_audio.numpy()
            batched_labels = batched_labels.numpy()

        return batched_audio, batched_labels