"""Builds the canonical file list for the primary dataset.

Deliberately excludes Positive/Labelled: it duplicates the same 422 photos
in Positive/Unlabelled but with bounding-box annotations drawn directly into
the pixels, which would let a model learn to detect the drawn boxes instead
of real cell morphology, and would also duplicate images across the folder
boundary. See docs/dataset_findings.md for the full reasoning.

Positive class = 1 = sickle-positive. Negative class = 0 = normal. This is
the single place that definition is made, so every downstream script
(splitting, training, evaluation) inherits it consistently rather than each
redefining it.
"""
from pathlib import Path

import pandas as pd

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0


def build_primary_manifest(train_source_root):
    """Returns a DataFrame with columns: path, label, label_name.

    train_source_root should point at the extracted primary dataset, i.e.
    the folder containing Positive/ and Negative/ subfolders.
    """
    root = Path(train_source_root)
    positive_dir = root / "Positive" / "Unlabelled"
    negative_dir = root / "Negative" / "Clear"

    rows = []
    for p in sorted(positive_dir.glob("*.jpg")):
        rows.append({"path": str(p), "label": POSITIVE_LABEL, "label_name": "positive"})
    for p in sorted(negative_dir.glob("*.jpg")):
        rows.append({"path": str(p), "label": NEGATIVE_LABEL, "label_name": "negative"})

    if not rows:
        raise FileNotFoundError(
            f"No images found under {positive_dir} or {negative_dir} - "
            "check that the primary dataset was downloaded and extracted "
            "(see notebooks/01_setup_and_inspect.ipynb)."
        )

    return pd.DataFrame(rows)
