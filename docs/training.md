# Training: Design Decisions

Documents `src/training/` and the notebooks that use it. Same spirit as
`docs/data_pipeline.md`: defaults chosen to keep moving, flagged clearly so
they can be revisited rather than silently locked in.

## Staged transfer learning, in plain language

Both models start from ImageNet-pretrained weights (they already know how
to recognize general shapes/edges/textures from millions of photos) and get
adapted to this task in stages, rather than retraining everything at once:

- **Stage 1 - head only:** freeze the entire pretrained backbone, train
  only the new classification layer we added on top. This is the safest
  starting point - with only ~570 images, retraining a multi-million
  parameter network from scratch would overfit badly, so we only teach the
  new final layer to interpret the backbone's existing features for this
  task.
- **Stage 2 - fine-tune the last block:** unfreeze the backbone's last
  major block (in addition to the head) and continue training at a lower
  learning rate. This lets the network adjust its most task-specific,
  late-stage features (which tend to encode more abstract/specific
  patterns) to sickle cell morphology specifically, while keeping the
  earlier, more general-purpose layers frozen.
- **Stage 3 - fine-tune more (not run automatically):** unfreeze one
  additional block back. Code supports this (`set_stage(..., stage=3)`),
  but it's only used if stage 2's validation results suggest the model
  would benefit from adapting more of the backbone - a judgment call made
  after looking at real results, not a step run by default.

**ResNet18 and MobileNetV2 have different internal structures**, so "the
last block" is architecture-specific, not identical:
- ResNet18: stage 2 unfreezes `layer4` (its last of four sequential
  stages); stage 3 additionally unfreezes `layer3`.
- MobileNetV2: stage 2 unfreezes the last 3 of its 19 sequential
  inverted-residual/conv blocks; stage 3 unfreezes the last 7.

This is an approximation of "as similar as possible across both models,"
not a perfect match - flagged explicitly rather than implied to be exact.

## Default hyperparameters (flagged, not locked in)

| Setting | Value | Why |
|---|---|---|
| Stage 1 epochs (max) | 15 | Head-only training converges fast; early stopping will usually cut this short |
| Stage 1 learning rate | 1e-3 | Only a small linear head is training - can afford a higher rate |
| Stage 2 epochs (max) | 15 | Same logic as stage 1 |
| Stage 2 learning rate | 1e-4 | Now touching pretrained backbone weights - a lower rate avoids destroying what it already learned |
| Batch size | 32 | Reasonable for a few hundred images per fold on a T4 GPU |
| Early stopping patience | 5 epochs | Stops if validation loss hasn't improved in 5 epochs; resets at each stage transition (see `src/training/loop.py`) so newly-unfrozen parameters get a fair chance, while the best-ever checkpoint is still tracked globally across stages |
| Random seed | 42 | For model init and data loader shuffling, alongside the data-split seeds already documented in `docs/data_pipeline.md` |

Same settings used for both models and both CV/final-training contexts,
per the project brief's "same major training conditions for both models
unless documented reason not to."

## Compute plan: CV first, final training as a separate, later step

5-fold CV x 2 models = 10 separate training runs before any final model
exists. Rather than one long notebook that also jumps straight into final
training, `notebooks/04_cross_validation.ipynb` does ONLY the CV comparison
and stops there for review - training the final models on the full
development set is a follow-up notebook, run only after CV results look
reasonable. This matches the project brief's requirement to stop and ask
before committing to the full computational plan if CV turns out to be
unreasonably expensive, and gives a natural checkpoint either way.

## What CV is/isn't used for here

Per `docs/data_pipeline.md`, the grouped 5-fold CV built in
`notebooks/03_data_pipeline.ipynb` compares model configurations - here,
that means comparing ResNet18 vs MobileNetV2 under the identical protocol
above. It is not a hyperparameter search (no per-fold tuning), and it does
not touch the held-out test set. Both models proceed to final training
regardless of which one "wins" CV - the brief wants both compared side by
side in the end, with trade-offs discussed rather than declaring a single
winner from CV alone.
