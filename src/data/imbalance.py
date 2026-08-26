"""Class-imbalance handling.

Chosen approach: class-weighted loss (inverse class frequency), computed
from the training data only. See docs/data_pipeline.md for why this was
picked over oversampling/undersampling - flagged there as a default choice,
not a silent one, in case a different approach is preferred later.
"""
import torch


def compute_class_weights(labels, num_classes=2):
    """Returns a torch.FloatTensor of per-class weights (inverse frequency,
    normalized so weights average to 1), suitable for passing as `weight=`
    to nn.CrossEntropyLoss. `labels` should be the labels of the TRAINING
    split only - never the val/test/CV-held-out portion.
    """
    labels = torch.as_tensor(list(labels), dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).float()
    if (counts == 0).any():
        raise ValueError(f"At least one class has zero training examples: {counts.tolist()}")
    inverse = 1.0 / counts
    weights = inverse * (num_classes / inverse.sum())
    return weights
