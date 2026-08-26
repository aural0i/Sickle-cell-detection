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
   (`kaggle.json`, from https://www.kaggle.com/settings/account) when prompted.
3. This downloads both datasets into separate folders and prints their
   structure, class counts, image properties, and any bundled license/README
   files, without training anything.

Further notebooks/scripts (data splitting, training, evaluation) will be added
here as the project progresses through its review checkpoints.

## Datasets

- **Primary:** Sickle Cell Disease Dataset (Tushabe et al.), via Kaggle
  (`florencetushabe/sickle-cell-disease-dataset`)
- **Potential external validation:** erythrocytesIDB, via Zenodo
  (record 18299474)

License terms and scientific compatibility for both are being verified before
any external validation claims are made - see `docs/` for findings once
available.

## Limitations

This project's full limitations section (data leakage risks, sample size,
class imbalance, dataset shift, cross-validation stability, and more) will be
documented in `docs/` once evaluation is complete, and summarized here.
