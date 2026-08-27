# Data Pipeline: Design Decisions

This documents the choices made while building `src/data/` and
`notebooks/03_data_pipeline.ipynb`. Several of these are defaults chosen to
keep the project moving rather than blocking on a decision that has more
than one reasonable answer - flagged clearly below so they can be revisited.

## Scope

Only the primary dataset trains and internally validates the model:
`Positive/Unlabelled` (422, positive/sickle) + `Negative/Clear` (147,
negative/normal) = 569 images. `Positive/Labelled` is excluded entirely (see
`dataset_findings.md`). erythrocytesIDB and Chula-RBC-12 are reserved for
evaluation only and are never touched by this pipeline.

**Positive class definition (used consistently everywhere from here on):**
label `1` = positive = sickle. label `0` = negative = normal. Set once in
`src/data/manifest.py` so every downstream script inherits it rather than
redefining it.

## Leakage prevention: duplicate-cluster grouping

No patient/slide/sample ID exists anywhere in the primary dataset (checked
in Step 1 - see `dataset_findings.md`). As a partial substitute,
`src/data/duplicates.py` detects exact duplicates (file hash) and near
duplicates (perceptual hash, default Hamming-distance threshold of 5) across
all 569 images, and every split/fold in `src/data/splitting.py` groups by
duplicate-cluster ID rather than treating images as independent. This
guarantees a duplicate or near-duplicate photo can never end up split across
train/val/test.

**This does not fully solve the missing-patient-ID problem.** It only
catches images that are literally or near-identically the same photo - it
cannot catch two *different* photos taken from the same patient's slide.
That residual leakage risk stays documented as an open limitation, not
resolved by this step.

### Open finding: cross-class duplicate groups (under investigation)

Running this against the real dataset (569 images) found 544 distinct
groups - 25 images cluster with at least one other image. Most of those
clusters are internally consistent (all-positive or all-negative), but
**6 groups contain both a positive- and a negative-labeled image**:
groups 34, 49, 92, 109, 119, and 131 (see the notebook output for exact
filenames).

This has two possible explanations with very different implications: (a) a
perceptual-hash false positive - different photos that hash similarly
because blood smear images are visually homogeneous overall, which would be
harmless, or (b) an actual labeling problem in the source Kaggle dataset -
the same photo present under both labels, which would be a real data
quality issue affecting confidence in the labels generally, not just these
few images.

**Resolved: (a), a perceptual-hash false positive - not a labeling
problem.** None of the 6 cross-class pairs were exact file matches, all
landed at exactly phash distance 4 (suspiciously uniform for supposedly
independent near-duplicates), and the user visually confirmed the paired
images are genuinely different photos, not the same picture twice. The
source dataset's labels are fine. The real issue was the near-duplicate
threshold (5) being too loose for this image domain - blood smear photos
share enough overall visual structure (similar staining, similar framing)
that phash doesn't discriminate between them as well as it does for
ordinary photos. **Fix confirmed working.** After tightening `DEFAULT_NEAR_DUP_THRESHOLD` from
5 to 2 in `src/data/duplicates.py` and re-running against the real dataset:
569 images -> 562 distinct groups (7 images in same-class-only clusters:
two negative/negative pairs, one negative/negative pair, and three
positive-only clusters), **0 cross-class groups**. Closed.

## Validation vs. cross-validation: how they relate

Defined once, used consistently, per the project brief's requirement not to
run two redundant selection procedures:

1. **Held-out test set** (`make_held_out_test_split`): one single carve-out,
   ~20% of the 569 images (grouped + stratified), touched exactly once, at
   final evaluation. Everything else is the "development set."
2. **Grouped 5-fold CV on the development set** (`make_cv_folds`): used
   ONLY to compare model configurations (ResNet18 vs MobileNetV2, or
   different training protocols). This is the actual model-selection
   mechanism - each fold's held-out portion serves as that config's
   validation data during comparison.
3. **Final train/val split on the full development set**
   (`make_final_train_val_split`, ~15% held out): used ONLY for early
   stopping / checkpoint selection when training the one chosen
   configuration on the full development set, after step 2 already picked
   it. This is a mechanical stopping signal, not a second model-selection
   step - no configuration decisions are made using this split.

Both (2) and (3) are grouped stratified splits (`StratifiedGroupKFold`),
automatically reducing the requested fold count (with an explanation
printed, never silently) if a class doesn't have enough distinct groups to
support it - this matched the primary dataset (422/147 split) comfortably
in testing, but the code doesn't assume that.

## Class imbalance: chosen approach (flagged, not silent)

~74% positive / 26% negative (ratio ≈ 2.9:1). **Chosen approach:
class-weighted loss** (`src/data/imbalance.py`, inverse-frequency weights
computed from the training data only, passed to `nn.CrossEntropyLoss`).

Why this over the alternatives:
- **Oversampling** the minority class would either duplicate existing
  negative images (risking the model memorizing repeats) or require
  synthetic generation, adding complexity without more real information.
- **Undersampling** the majority class would throw away already-scarce
  data - down to 147/147 = 294 total images, a big cut from 569.
- **Class-weighted loss** uses every real image exactly once per epoch and
  just changes how much each class's errors count toward the loss - simplest
  option that doesn't shrink or duplicate the dataset.

This is a default, not a locked-in decision - happy to switch to a weighted
sampler or a different approach if there's a reason to prefer one.

## Augmentation: why each one is biologically reasonable (train split only)

- **Small rotations (±15°):** a blood smear's orientation under the
  microscope is a function of how the slide was placed, not anything
  biological - rotating the image doesn't change what it depicts.
- **Horizontal AND vertical flips:** there's no canonical "up" for cells on
  a slide (unlike, say, a photo of a person, where orientation carries
  meaning). Mirroring doesn't produce biologically implausible cell shapes -
  sickle-shaped and normal cells remain valid shapes under reflection.
- **Brightness/contrast jitter (moderate, ±20%):** staining intensity and
  microscope illumination genuinely vary between slides and photos in real
  practice - this mimics that real technical variation.
- **Deliberately NOT doing hue/color jitter:** Field and Leishman staining
  produce a characteristic color palette that could itself carry real
  diagnostic signal. Randomizing hue risks either destroying that signal or
  making the augmented image resemble a different stain type entirely -
  not a realistic or safe augmentation here.

Applied only when `train=True` (`src/data/dataset.py`) - validation, the
held-out test set, and both external validation datasets never see
augmentation, per the project brief.

## Reproducibility

Every split/fold function takes an explicit `seed` parameter (defaults:
42 for the held-out test split, 43 for CV folds, 44 for the final
train/val split - documented here rather than buried in code). Given the
same downloaded dataset and the same seeds, the split is 100% deterministic
- that determinism, not a single committed CSV, is the actual reproducibility
mechanism, since this dev environment can't run the notebook itself to
produce that CSV (no GPU, no dataset access - see `environment.md`). The
notebook saves the resulting split assignment to
`results/splits/primary_dataset_splits.csv` when run in Colab; if useful as
a committed artifact, send that file back and it can be added to the repo.

## Lessons learned: pip install hangs in Colab

While running notebook 03, a `pip install -q -r <filtered requirements.txt>`
cell ran for 20+ minutes without finishing (user reported it, correctly
flagged as not normal). Root cause: `requirements.txt` hard-pinned exact
versions of packages (numpy, pandas, Pillow, scikit-learn, scikit-image,
matplotlib, seaborn, tqdm) that Colab already ships - at different,
newer versions - as part of its base image. Forcing pip to reconcile old
exact pins against a large pre-existing environment sends its dependency
resolver into extensive backtracking, which can run far longer than 20
minutes or effectively never finish. This is the same underlying issue as
the earlier `torchvision==0.20.1` install failure, just recurring across a
wider set of packages once the notebook tried to install "everything except
torch/torchvision" instead of "everything except what Colab already has."

**Fix:** notebooks 01 and 03 now install only the packages Colab genuinely
doesn't ship by default (`kaggle`, and `imagehash` for notebook 03), with no
version pins, instead of installing from `requirements.txt` at all inside
Colab. `requirements.txt` itself was also changed from exact `==` pins to
`>=` floors throughout, both to reflect that Colab's own versions are fine
and to reduce (not eliminate) this same risk for anyone installing it on a
from-scratch, non-Colab machine.

**If a future notebook needs a new package:** add it directly to that
notebook's install cell (unpinned, e.g. `!pip install -q grad-cam`) rather
than pointing at `requirements.txt` - and only add it to `requirements.txt`
for the from-scratch/non-Colab reproducibility story.
