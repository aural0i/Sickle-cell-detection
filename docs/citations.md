# Citations & Attribution

This project uses data from other researchers' published work. Anywhere this
project's results, README, or portfolio write-up references a dataset, it
should credit the dataset using the citation below - not just link to it.

## Primary training dataset

Tushabe, F., Mwesige, S., Kasule, V., Nsiimire, E., Musani, S. C., Areu, D.,
Mutabazi, P., & Othieno, E. Sickle Cell Disease Dataset. Kaggle.
https://www.kaggle.com/datasets/florencetushabe/sickle-cell-disease-dataset

Related publications (per the dataset's own description):
- Tushabe, F. et al. "A Dataset of Microscopic Images of Sickle Cells."
  Presented at the 5th Global Webinar on AI, ML, Data Science & Robotics,
  April 15-16, 2024.
- Tushabe, F. B., Mwesige, S., Kasule, V., Nsiimire, E., Musani, S. C., et
  al. (2025). "An Image-Based Sickle Cell Detection Method." Engineering
  and Applied Sciences Journal, 2(1), 1-4.
- Tushabe, F., et al. "A Dataset of Microscopic Images of Sickle and Normal
  Red Blood Cells." Acta Scientific Microbiology, 7(12) (2024): 22-29.

License: listed by Kaggle as `DbCL-1.0` (Open Data Commons Database
Contents License). Exact terms not independently verified by Claude (see
`dataset_findings.md`) - please confirm the license text shown on the
Kaggle page itself before any public/portfolio use beyond what's documented
here.

## External validation dataset (compatibility limitations documented separately)

Marrero Fernández, P. D., Coello Said, G., Delgado Font, W. E., Herold
Garcia, S., Fernández García, K., Montoya Padrón, A., González-Hidalgo, M.,
& Jaume-i-Capó, A. erythrocytesIDB (Version 2, October 2017). Zenodo.
https://doi.org/10.5281/zenodo.18299474

License: CC BY-NC-ND 4.0 (Attribution-NonCommercial-NoDerivatives). See
`dataset_findings.md` for the resulting usage policy (aggregate metrics
only, no public display of images from this dataset) and for why this
dataset could not directly serve as external validation for this project's
sickle-vs-normal task.

## Healthy-comparison candidate dataset (under evaluation)

Naruenatthanaset, K., Chalidabhongse, T. H., Palasuwan, D.,
Anantrasirichai, N., & Palasuwan, A. (2021). "Red Blood Cell Segmentation
with Overlapping Cell Separation and Classification on Imbalanced
Dataset." arXiv:2012.01321. Dataset: Chula-RBC-12-Dataset, Zenodo.
https://zenodo.org/records/5638201

License: dataset's own Zenodo listing is `"Other (Open)"` with no further
description (confirmed directly from the Zenodo page - not the same as the
dataset's GitHub code repository, which is MIT). See `dataset_findings.md`
for the resulting conservative usage policy (research use + citation, no
redistribution of raw files, no public display of images from this
dataset without further clarification).

## General note

If any dataset's license or terms are updated or clarified after this was
written, update both this file and `dataset_findings.md` together so they
stay consistent.
