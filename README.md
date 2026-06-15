# Dentin-Normalized CBCT Radiomics for Differential Diagnosis

This repository contains code for dentin-reference intensity normalization, PyRadiomics feature extraction, and machine-learning analysis for differential diagnosis of periapical granulomas and radicular cysts from CBCT images.

The repository is intended for code availability. The original CBCT images, segmentation masks, and extracted feature tables are not publicly distributed because they contain protected patient information and are subject to institutional data-sharing restrictions.

## Repository structure

```text
configs/
  pyradiomics_original_features.yaml

data/
  manifest_template.csv

notebooks/
  01_dentin_reference_roi_and_normalization.ipynb
  02_radiomics_feature_extraction.ipynb
  03_machine_learning_analysis.ipynb

src/
  dentin_reference_normalization.py
  radiomics_extraction.py
  ml_analysis.py
  delong.py
```

## Label convention

The preprocessing code assumes the following segmentation labels:

| Label | Structure |
|---:|---|
| 1 | Lesion |
| 2 | Radiopaque material or restoration |
| 3 | Bone |
| 4 | Tooth structure |

The machine-learning code uses:

| Class | Diagnosis |
|---:|---|
| 0 | Periapical granuloma |
| 1 | Radicular cyst |

## Workflow

1. Prepare a manifest CSV with `case_id`, `image_path`, `label_path`, `reference_cohort`, and optionally `diagnosis`.
2. Run `01_dentin_reference_roi_and_normalization.ipynb` to estimate the dentin reference distribution and normalize CBCT images.
3. Run `02_radiomics_feature_extraction.ipynb` for raw and dentin-normalized images.
4. Run `03_machine_learning_analysis.ipynb` to compare raw and dentin-normalized radiomic feature sets.

## Methodological notes

- The dentin reference cohort should be fixed before model evaluation and should not be selected using diagnosis labels or model outcomes.
- PyRadiomics images and masks are read with SimpleITK to preserve voxel spacing and image metadata.
- No missing-value imputation is applied. Missing feature values are treated as a preprocessing error.
- Feature screening follows the manuscript-described procedure: Levene's test, followed by Student or Welch two-sample t-test, with feature retention at `p < 0.05`.
- Feature screening, scaling, SMOTE, and hyperparameter tuning are performed inside training folds to avoid leakage.
- PyRadiomics discretization is set to `binWidth: 100`, matching the original analysis notebook setting. Change this only if the manuscript reports a different bin width.
