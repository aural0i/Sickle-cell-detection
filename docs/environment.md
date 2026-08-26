# Step 0: Environment & Compute Check

Date checked: 2026-08-26

## What this session's machine actually has

| Resource | Finding |
|---|---|
| CPU | 4 cores (Intel Xeon @ 2.10GHz) |
| RAM | 15 GB |
| GPU | None detected (no `nvidia-smi`, no CUDA toolkit, no VGA GPU device) |
| Disk | ~30 GB writable allowance |
| Network egress | Blocked by session policy to `kaggle.com` and `zenodo.org` (HTTP 403 from the egress proxy on both) |

## Verdict: this remote session is not adequate for this project

Two independent blockers, either one alone would be disqualifying:

1. **No GPU.** Training two CNNs (ResNet18 + MobileNetV2) with transfer learning,
   5-fold grouped cross-validation (10 total training runs minimum before even
   counting the final full-development-set models), plus Grad-CAM, is designed
   for GPU acceleration. On 4 CPU cores this would realistically take many hours
   to multiple days depending on dataset size, and this remote session is an
   ephemeral container that can be reclaimed - a multi-day CPU job is not a
   realistic or safe plan here.
2. **No network access to the data sources.** This session's outbound network
   policy blocks `kaggle.com` and `zenodo.org` directly (403 from the egress
   proxy). That means the Kaggle API download and the Zenodo download specified
   in the task cannot run inside this session at all, independent of compute
   power.

## Recommendation

Do the actual data download + GPU training in **Google Colab (free tier)**,
which provides:
- Free NVIDIA T4 GPU access (sufficient for ResNet18/MobileNetV2 fine-tuning
  at typical microscopy-crop image sizes)
- Native, unrestricted access to the Kaggle API (with an uploaded `kaggle.json`
  token) and to Zenodo
- A notebook environment well suited to the "explain each step, check in before
  big decisions" workflow this project calls for

**Plan going forward:**
- This repository (code, docs, results, README) is developed and version
  controlled here as normal.
- The actual heavy lifting - downloading data, training, cross-validation,
  Grad-CAM - is written as scripts/notebooks in `/src/` that are designed to be
  run in Colab (or any machine with a GPU and open network access), not
  auto-executed in this session.
- Anything that is cheap and doesn't need the datasets or a GPU (project
  scaffolding, dependency pinning, writing the data pipeline code, writing the
  README/docs) is still done directly in this session.
- Before any step that would need to actually execute in Colab, I will hand you
  a specific notebook/script and tell you what to run and what to expect,
  rather than silently assuming it succeeded.

## Reproducible environment setup

See `requirements.txt` at the project root. Recommended setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In Colab, the same `requirements.txt` can be installed with
`!pip install -r requirements.txt` in the first cell (Colab already ships
compatible CUDA drivers, so the pinned `torch`/`torchvision` wheels will pick
up GPU support automatically).

## Compute/cost flags for the rest of the project

Per the project instructions, here is what to expect will be the heavier steps,
flagged in advance:

- **Full dataset download** (Kaggle + Zenodo): bandwidth/time cost depends on
  dataset size (unknown until Step 1 inspection) - flagged, will confirm sizes
  before downloading.
- **5-fold grouped cross-validation x 2 models**: the single heaviest compute
  step in the project. Even on a Colab T4 this could be tens of minutes to a
  few hours depending on dataset size and epoch budget - will propose a
  specific epoch/fold budget before running it and get your sign-off.
- **Grad-CAM generation**: cheap relative to training (inference-only, small
  number of images), not a concern.
