import numpy as np

def top_k_accuracy(
    y_true,
    y_score,
    k: int =3,
    normalize: bool=True,
    default_score: float | None = 0.5,
) -> float:
    """
    Compute top-k multilabel accuracy, *ignoring* any class whose score
    is the same in every sample.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, n_classes)
        Binary indicator of true labels.

    y_score : array-like, shape (n_samples, n_classes)
        Score for each class.

    k : int, default=3
        Number of top predictions to consider.

    normalize : bool, default=True
        If True, returns fraction of samples with at least one hit.
        Else returns count of such samples.

    default_score : float or None, default=0.5
        If not None, any class (column) whose scores are all equal to
        default_score (up to atol) will be dropped before computing.

    Returns
    -------
    score : float
        The top-k multilabel accuracy score.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"Shape mismatch: y_true{y_true.shape} vs y_score{y_score.shape}"
        )

    if default_score is not None:
        # mask[j] = True if class j is *not* all-default
        mask = ~np.all(
            np.isclose(y_score, default_score),
            axis=0
        )
        if not np.any(mask):
            raise ValueError(
                "All classes have default_score; nothing to evaluate."
            )
        y_true = y_true[:, mask]
        y_score = y_score[:, mask]

    n_samples, n_classes = y_true.shape
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    k = min(k, n_classes)

    # 2) Find top-k scoring class‐indices per sample (unsorted within top-k)
    topk_idx = np.argpartition(y_score, -k, axis=1)[:, -k:]
    rows = np.arange(n_samples)[:, None]

    # 3) Check if any true label appears in those top-k positions
    hits = np.any(y_true.astype(bool)[rows, topk_idx], axis=1)
    correct = hits.sum()

    return (correct / n_samples) if normalize else int(correct)