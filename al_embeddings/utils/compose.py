import torch

class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below. Implementation from torchvision.

    Args:
        transforms (list of ``callable`` objects): list of callables to chain.

    Example:
        >>> transforms.Compose([
        >>>     lambda x: x[-1],
        >>>     lambda x: torch.tensor(x),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential()
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, batch):
        for t in self.transforms:
            batch = t(batch)
        return batch

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string