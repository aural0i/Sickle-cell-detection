# Sickle Cell Detection - CNN Research/Portfolio Project

> **This is an experimental research and portfolio project demonstrating machine
> learning methodology. It is NOT a medical diagnostic device.** It does not
> diagnose sickle cell disease and must never be used as a substitute for
> laboratory testing (e.g. hemoglobin electrophoresis) or a qualified medical
> professional. See [`docs/`](docs/) for full details as the project develops.

## Status

This project is being built step by step, with checkpoints for review before
each major stage (data pipeline, training, evaluation). See
[`docs/environment.md`](docs/environment.md) for the Step 0 compute/network
findings and why data download + training run in Google Colab rather than in
the automated development session.

## Project layout

```
data/         # datasets (not committed - see .gitignore). train_source/ and
              # external_val/ are always kept separate and never mixed.
notebooks/    # Colab notebooks for data download, inspection, and training
src/          # reusable pipeline code (splitting, dataset classes, training, eval)
models/       # trained model checkpoints (not committed - see .gitignore)
results/      # metrics, figures, confusion matrices, CV results, error analysis
docs/         # environment notes, methodology writeups, this project's docs
```

## Reproducing the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

1. Open [`notebooks/01_setup_and_inspect.ipynb`](notebooks/01_setup_and_inspect.ipynb)
   in Google Colab (`Runtime > Change runtime type > GPU`).
2. Run the cells top to bottom. You'll need a Kaggle API token
   (the `KGAT_...` kind, from https://www.kaggle.com/settings/api) when prompted.
3. This downloads the primary and external-validation datasets into separate
   folders and prints their structure, class counts, image properties, and
   any bundled license/README files, without training anything.
4. Separately, [`notebooks/02_healthy_comparison_chula_rbc12.ipynb`](notebooks/02_healthy_comparison_chula_rbc12.ipynb)
   is a standalone notebook (no GPU needed, no dependency on notebook 01)
   that downloads and inspects a candidate healthy-comparison dataset.
5. [`notebooks/03_data_pipeline.ipynb`](notebooks/03_data_pipeline.ipynb)
   (also standalone, no GPU needed) builds the leakage-safe train/CV/test
   split for the primary dataset and demonstrates the augmentation
   pipeline - see [`docs/data_pipeline.md`](docs/data_pipeline.md) for the
   full reasoning behind the split strategy, class-imbalance handling, and
   why each augmentation is biologically reasonable.

Further notebooks/scripts (training, evaluation) will be added here as the
project progresses through its review checkpoints.

## Datasets

- **Primary:** Sickle Cell Disease Dataset (Tushabe et al.), via Kaggle
  (`florencetushabe/sickle-cell-disease-dataset`)
- **Potential external validation:** erythrocytesIDB, via Zenodo
  (record 18299474) - found not to have a healthy/normal comparison group
  (see `docs/dataset_findings.md`)
- **Candidate healthy-comparison dataset:** Chula-RBC-12-Dataset
  (Naruenatthanaset et al., arXiv:2012.01321), via Zenodo (record 5638201) -
  under evaluation

License terms and scientific compatibility for all datasets are being
verified before any claims are made based on them - see
[`docs/dataset_findings.md`](docs/dataset_findings.md) for current findings
and [`docs/citations.md`](docs/citations.md) for full citations.

## Data licensing policy

Not every dataset used here comes with a fully clear, unambiguous license.
Rather than assume the most permissive reading, this project follows one
consistent rule for any dataset whose redistribution/derivative-work rights
aren't clearly spelled out:

- Using it for training/internal evaluation and reporting aggregate metrics
  (accuracy, sensitivity, etc.) is fine.
- Publicly displaying raw or derived images from that dataset (example
  crops, error-analysis figures, Grad-CAM overlays) is not, unless its
  license clearly permits derivative works.
- The dataset is always cited properly regardless of its license status
  (see `docs/citations.md`).
- Nothing under `data/` is ever committed to this repo for any dataset -
  only fetched on demand via the notebooks/scripts here - which keeps
  redistribution out of the question entirely.

This currently applies to erythrocytesIDB (CC BY-NC-ND - no derivatives)
and the Chula-RBC-12-Dataset (Zenodo license listed only as `"Other (Open)"`,
no further terms given). See `docs/dataset_findings.md` for the specifics
behind each.

## Limitations

This project's full limitations section (data leakage risks, sample size,
class imbalance, dataset shift, cross-validation stability, and more) will be
documented in `docs/` once evaluation is complete, and summarized here.
