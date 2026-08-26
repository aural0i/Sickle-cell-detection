# Step 1: Dataset Inspection Findings

Generated from the output of `notebooks/01_setup_and_inspect.ipynb`, run in
Google Colab (Tesla T4, 15 GB VRAM - confirms Step 0's compute plan works).

## Bug found and fixed during this run

The notebook's dependency-install cell tried to pin `torchvision==0.20.1`,
which no longer exists on PyPI. Pip failed loudly for that one package but
the cell didn't stop, so it silently fell back to Colab's preinstalled
`torch`/`torchvision`. Fixed by no longer trying to reinstall those two in
Colab at all (Colab's preinstalled pair is already GPU-matched); everything
else still installs from `requirements.txt` as intended. See the updated
`requirements.txt` and notebook.

## Primary dataset: Sickle Cell Disease Dataset (Tushabe et al., Kaggle)

**What it is:** microscopy images of stained peripheral blood smears from
140 patients at hospitals in the Teso region of Uganda (Kumi Hospital, Soroti
Regional Referral Hospital, Soroti University), imaged after Field or
Leishman staining. This is genuine, appropriately-sourced microscopy data for
this task.

**License:** listed by Kaggle as `DbCL-1.0` (Open Data Commons Database
Contents License). **I could not independently verify the exact legal text**
- opendatacommons.org, Wikipedia, and Kaggle itself are all network-blocked
from this development session, and it's not in SPDX's standard license list.
Please open the dataset page yourself
(https://www.kaggle.com/datasets/florencetushabe/sickle-cell-disease-dataset)
and check the exact license text/wording shown there (attribution
requirements, commercial-use terms) before this is used in a public
portfolio, and paste it back if you want me to review it.

**Folder structure found:**
```
data/train_source/
  Positive/
    Labelled/     422 images  (positive = sickle cell)
    Unlabelled/   422 images  (positive = sickle cell)
  Negative/
    Clear/        147 images  (negative = normal)
```
The dataset description also mentions 122 "Not clear" negative images and a
larger unclear-images set, but those were not included in the actual
download ("Due to space restrictions, we have not uploaded these images").
That matches what we see - not a download error.

### Important finding: `Positive/Labelled` and `Positive/Unlabelled` are the same 422 photos, twice

The dataset's own description confirms this: *"The folder labelled contains
the positive images with bounding boxes around the seen sickle cells. The
folder UnLabelled, are the positive images unlabelled."* Same file-count
(422 in each), same naming convention (`1.jpg`...`422.jpg` in both).

**This matters a lot:** the `Labelled` copies have annotation marks
(bounding boxes) drawn directly into the image pixels. If we used both
copies, two problems appear at once:
1. **Leakage** - the same underlying photo would exist in both folders, so a
   naive split could put one copy in training and the other in the test set,
   letting the model "cheat" on a photo it already memorized.
2. **Shortcut learning** - a CNN could learn to detect drawn bounding-box
   pixels rather than actual cell morphology, since only the positive class
   has them.

**Recommendation: use only `Positive/Unlabelled` + `Negative/Clear` for
training/evaluation, and exclude `Positive/Labelled` entirely from the
classification pipeline.** That gives a clean, unannotated set:

| Class | Folder | Count |
|---|---|---|
| Positive (sickle) | `Positive/Unlabelled` | 422 |
| Negative (normal) | `Negative/Clear` | 147 |
| **Total** | | **569** |

**Class imbalance:** ~74% positive / 26% negative (ratio ≈ 2.9:1). This is
substantial and will need addressing (class-weighted loss and/or
sensitivity/specificity/PR-AUC reporting rather than relying on accuracy) -
to be decided explicitly at the Data Inspection stage, not silently.

**Image properties:** all `.jpg`, all RGB, roughly 1000x1000 px (varies
slightly per image - typical of individually captured microscope fields, not
a single uniform sensor crop).

### Data leakage prevention: no patient/slide ID grouping is available

We looked for a metadata file, README, or patient-ID pattern in filenames -
found none. Files are just sequentially numbered (`1.jpg`, `2.jpg`, ...)
within each class folder, with no indication of which of the 140 patients
each image came from. **This is a real limitation**: we cannot guarantee
that images from the same patient/slide won't end up split across
train/val/test. We will document this explicitly and treat images as
independent samples, flagged as an unverified leakage risk in the final
write-up. (One option, if you want to pursue it: the dataset description
lists a contact email, floratush@gmail.com, for inquiries - the authors
might have patient-level metadata available on request.)

## External dataset: erythrocytesIDB (Zenodo record 18299474)

**What it is:** peripheral blood smear images from Sickle Cell Disease
patients at a hospital in Santiago de Cuba, Giemsa-stained, imaged at 100x
with a consumer camera (not a professional microscopy camera setup).
Three sub-collections:
- **erythrocytesIDB1**: 196 full-field images, plus 629 individually cropped
  cells labeled by **shape**: circular / elongated / other.
- **erythrocytesIDB2**: 50 full-field images, each with a full annotation
  mask set (overall cell mask, circular-cell mask, elongated-cell mask,
  other-cell mask).
- **erythrocytesIDB3**: 30 full-field images, same mask structure as IDB2.

**License:** `CC BY-NC-ND 4.0` (Attribution-NonCommercial-NoDerivatives).
This is one of the most restrictive Creative Commons licenses:
- **NonCommercial** - fine for a non-commercial portfolio project.
- **NoDerivatives** - this is the concerning one. It prohibits distributing
  *modified* versions of the images. Showing processed crops, augmented
  versions, or Grad-CAM overlays generated from these specific images in a
  public portfolio (README, results gallery, etc.) would plausibly count as
  a derivative and should be avoided. Reporting aggregate *numbers* (e.g.
  "accuracy on external validation set: X%") without displaying the
  underlying images is a much safer use of this license.

**Scientific compatibility with our task: NOT a direct match.** Several
mismatches:

1. **Different task/label semantics.** Our primary dataset's label is
   image-level "sickle-positive vs. normal" (does this blood sample show
   sickle cell disease or not). erythrocytesIDB's own label taxonomy is
   individual-**cell shape**: circular / elongated / other. "Elongated" is
   a commonly-used *morphological proxy* for sickle-shaped cells in the
   image-processing literature, but it is not the same labeled construct as
   our primary dataset's diagnosis-oriented label.
2. **No apparent healthy/negative population.** The dataset description
   states all images come from "patients with Sickle Cell Disease" at a
   Special Hematology Department. There's no separate "normal/healthy
   donor" class visible anywhere in the folder structure or description -
   which means there may be no clean "negative" group to validate
   specificity against at all.
3. **Different unit of analysis.** Our primary dataset labels whole
   microscopy-field images. erythrocytesIDB1's shape labels are on
   individually-cropped single cells - a much smaller, tighter crop than a
   full field.
4. **Mixed file types bundled together.** Each per-image folder in IDB2/IDB3
   contains the original photo (`source.jpg`) *and* several derived
   segmentation masks (`mask.jpg`, `mask-circular.jpg`, etc.) and an
   annotated overlay (`labeled.jpg`) - all in the same folder. Our
   inspection script's image-summary counted all of these together, so the
   "grayscale vs RGB" numbers it reported are a mix of real photos and
   binary/grayscale masks, not directly meaningful yet. Using this data
   would require carefully picking out only `source.jpg` files and ignoring
   the masks/overlays for a classification task.
5. **Junk files from the zip.** The extracted archive also contains a
   `__MACOSX/` folder and `.DS_Store` files - these are macOS packaging
   artifacts, not data, and must be filtered out of anything we process.

**Conclusion: erythrocytesIDB cannot support a direct, like-for-like
"sickle vs. normal" external validation** the way the primary dataset's task
is framed - there's no clear normal/negative population in it, and the
labeling scheme measures something related but different (cell shape, not
diagnosis).

### Options going forward (needs your decision)

- **A - Skip external validation.** Report only internal held-out test +
  cross-validation results on the primary dataset. Scientifically honest,
  but no external check on generalization.
- **B - Repurpose erythrocytesIDB1 as a distribution-shift stress test**,
  explicitly labeled as such (not a clinical validation): take its
  individually-cropped cells, treat "elongated" as an approximate proxy for
  "sickle-shaped" and "circular"/"other" as an approximate proxy for
  "not sickle-shaped," and see whether our model - trained on whole-field
  images from a different country, stain, and camera - still separates
  these differently-sourced, differently-cropped, differently-labeled images
  in a sensible direction. Report this with heavy caveats (proxy labels, no
  healthy population, different unit of analysis) rather than as a real
  validation metric.
- **C - Look for a better external validation dataset** with genuine
  image-level sickle/normal labels comparable to the primary dataset. None
  identified yet; would need a fresh search.

I'd lean toward **B** if you want some external signal at all, clearly
labeled as exploratory - but this is exactly the kind of call the project
brief said should come back to you rather than being decided silently.

## Candidate healthy-comparison dataset: Acevedo et al. (Mendeley Data)

You correctly pointed out that neither option above actually solves the
deeper problem: erythrocytesIDB has no healthy/normal-donor population at
all, so external validation can't properly test specificity (the "this is a
normal patient" side of the task) no matter how we slice it. You suggested
checking https://data.mendeley.com/datasets/snkd93bnjr/1 (Acevedo et al.,
"A dataset of microscopic peripheral blood cell images for development of
automatic recognition systems," *Data in Brief*, 2020) as a possible source
of isolated normal red blood cell images.

**Status: unverified, pending your Colab run.** `data.mendeley.com` is
blocked by this development session's network policy, the same as Kaggle and
Zenodo were, so I could not inspect it directly. From memory - not confirmed
- I recall this dataset's labeled classes are mostly white blood cell types
(neutrophils, eosinophils, basophils, lymphocytes, monocytes, immature
granulocytes) plus platelets plus erythroblasts (immature, nucleated red
cell precursors - not mature biconcave red blood cells), which would mean it
may *not* contain a dedicated normal-mature-erythrocyte class. This is only
a recollection and could be wrong.

Added Section 8 to the Colab notebook to check this for real, using Colab's
own (unrestricted) internet access rather than guessing further. It attempts
an automated check of the dataset page, with manual fallback instructions if
the page doesn't expose enough in raw HTML. Flagged in the notebook as a
larger download (~1-2 GB) worth confirming is worthwhile before running.

**Outcome: dead end, but not our network policy this time.** The automated
check hit Cloudflare's bot-protection challenge page ("Just a moment...",
HTTP 403) - `data.mendeley.com` screens out non-browser requests generally,
which would affect Colab too, not just this dev session. Not worth building
a browser-automation workaround for a single metadata check. Left the
Section 8 cells in the notebook as a documented dead end rather than
deleting them.

## Third candidate: Chula-RBC-12-Dataset (Zenodo, user-sourced)

User found and proposed **Chula-RBC-12-Dataset** (Naruenatthanaset et al.,
"Red Blood Cell Segmentation with Overlapping Cell Separation and
Classification on Imbalanced Dataset," arXiv:2012.01321, 2021), hosted at
https://zenodo.org/records/5638201. Per the dataset's own page (confirmed by
the user, not fetched by Claude - `zenodo.org` is still blocked here): 706
whole blood-smear images (640x480), >20,000 individually labeled RBCs across
12 shape classes, with **class 0 explicitly defined as "Normal cell"** - a
real healthy-comparison label, not a proxy this time. ~58 MB download.

**License status: partially confirmed, needs the user's eyes on one more
detail.** The dataset's GitHub repo (`Chula-PIC-Lab/Chula-RBC-12-Dataset`)
has an MIT License file (confirmed via `raw.githubusercontent.com`, which is
reachable from this dev session unlike Kaggle/Zenodo's own pages) - MIT is
permissive: use, modify, and redistribute freely, just keep the copyright
and license notice. **However**, Zenodo lists the *dataset's* license
separately as `"Other (Open)"`, which is not necessarily the same scope as
the code repo's MIT license. Claude could not load the Zenodo page directly
to check whether it states the same MIT terms for the actual image/label
files. **Open item: user to confirm the exact wording of the Zenodo page's
"License" section.**

**Citation requirement:** the dataset requires citing the arXiv paper above
if used - to be included in the project's README/references regardless of
final licensing outcome.

**Status: download + inspection added to the notebook (Section 9), not yet
run.** Per the user's explicit instruction, this only downloads and inspects
the label format for now - no extraction of "Normal cell" crops/coordinates
until we've actually seen the annotation file format (the GitHub README
only says each label line is "x coordinate, y coordinate, type of RBC in
number," without specifying the file format - YOLO-style txt, XML, JSON, or
CSV - or whether it's one file per image or one master file). Waiting on the
Colab run's output before writing any extraction code.
