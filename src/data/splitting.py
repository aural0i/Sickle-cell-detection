"""Leakage-safe splitting for the primary dataset.

Three distinct operations, deliberately kept separate rather than reused
from one another, so each one's role is unambiguous:

1. make_held_out_test_split - one-time carve-out of the untouched internal
   test set. Touch this exactly once, at final evaluation.
2. make_cv_folds - grouped stratified k-fold assignment on the development
   set (train+val combined), used ONLY to compare model configurations
   (e.g. ResNet18 vs MobileNetV2, or different training protocols).
3. make_final_train_val_split - a small validation slice carved out of the
   full development set, used ONLY for early stopping / checkpoint
   selection when training the FINAL chosen configuration on the full
   development set. This is not a second model-selection step - the model
   architecture/protocol was already chosen via CV; this split's only job
   is to say "stop training now" and "save this checkpoint."

All three are grouped by `group_col` (see duplicates.py) so a duplicate or
near-duplicate image can never land in two different splits/folds at once,
and stratified by `label_col` so class balance is preserved as closely as
possible in every split.
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _n_splits_for(df, group_col, label_col, requested_n_splits):
    """Reduces the requested fold count if there aren't enough groups per
    class to support it, and explains why, instead of forcing it.
    """
    min_groups_per_class = (
        df.groupby(label_col)[group_col].nunique().min()
    )
    n_splits = min(requested_n_splits, int(min_groups_per_class))
    if n_splits < requested_n_splits:
        print(
            f"Requested {requested_n_splits} splits, but the smallest class "
            f"only has {min_groups_per_class} unique group(s) - reducing to "
            f"{n_splits} splits instead of forcing an unsupported fold count."
        )
    return max(n_splits, 2)


def make_held_out_test_split(df, group_col="group_id", label_col="label",
                              n_splits=5, seed=42):
    """Carves off ~1/n_splits of df as a held-out test set.

    Returns (dev_df, test_df). n_splits=5 means a ~20% test set by default;
    reduced automatically (with an explanation printed) if there aren't
    enough distinct groups per class to support that.
    """
    n_splits = _n_splits_for(df, group_col, label_col, n_splits)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(sgkf.split(df, df[label_col], df[group_col]))
    dev_df = df.iloc[dev_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return dev_df, test_df


def make_cv_folds(dev_df, group_col="group_id", label_col="label",
                   n_splits=5, seed=43):
    """Adds a `cv_fold` column (0..n_splits-1) to a copy of dev_df, for
    grouped stratified cross-validation used to compare model
    configurations. Never touches the held-out test set.
    """
    n_splits = _n_splits_for(dev_df, group_col, label_col, n_splits)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_assignment = pd.Series(-1, index=dev_df.index)
    for fold_i, (_, val_idx) in enumerate(
        sgkf.split(dev_df, dev_df[label_col], dev_df[group_col])
    ):
        fold_assignment.iloc[val_idx] = fold_i
    out = dev_df.copy()
    out["cv_fold"] = fold_assignment.values
    out.attrs["n_cv_folds"] = n_splits
    return out


def make_final_train_val_split(dev_df, group_col="group_id", label_col="label",
                                val_frac=0.15, seed=44):
    """Adds a `final_split` column ('train' / 'val') to a copy of dev_df,
    for early stopping when training the final chosen configuration on the
    full development set. NOT used for model/config selection.
    """
    requested_n_splits = max(2, round(1 / val_frac))
    n_splits = _n_splits_for(dev_df, group_col, label_col, requested_n_splits)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf.split(dev_df, dev_df[label_col], dev_df[group_col]))
    out = dev_df.copy()
    out["final_split"] = "train"
    out.iloc[val_idx, out.columns.get_loc("final_split")] = "val"
    return out
