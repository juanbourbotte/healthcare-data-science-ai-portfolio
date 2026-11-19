# SHARE cognitive-status classification

Reproducible Python workflow for exploratory documentation and supervised
classification using an authorized, locally prepared SHARE-derived dataset.

## Research scope

The current target, `Y_deterioro`, is described in the original analysis as a
binary label derived from the lower quartile of the global cognitive score
(`SGC`). This script excludes `SGC` from predictors.

**Interpretation limitation:** if the label is defined from the lower quartile
of SGC measured at the same interview, the task is cross-sectional
classification of an operational cognitive-performance group. It is not, by
itself, prediction of future cognitive decline, diagnosis of dementia, or
clinical risk estimation.

Before publication, verify the target formula, cut-off, direction, ties,
missing-data rules, survey wave, sampling design, and SHARE codebook definitions.
Also audit every candidate predictor for direct or indirect overlap with the
items used to construct SGC.

## Repository structure

```text
share-cognitive-risk/
├── data/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── src/
│   └── main.py
├── .gitignore
├── README.md
└── requirements.txt
```

The SHARE-derived data file is intentionally not included. Do not upload
participant-level data, identifiers, restricted extracts, or files whose
license/terms do not explicitly permit redistribution.

## Requirements

- Python 3.10 or newer
- VS Code with the Python extension
- An authorized local analysis table saved as `data/df_final_listo.csv`

## Setup in VS Code

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/main.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/main.py
```

If your dataset has a different filename or location, edit `DATA_PATH` in
`src/main.py`.

## Outputs

The script writes results to `outputs/`:

- `missingness_summary.csv`
- `target_distribution.csv`
- `hyperparameter_search.csv`
- `test_metrics.json`
- `test_predictions.csv`
- `test_performance_curves.png`
- `permutation_importance.csv`
- `random_forest_pipeline.joblib`
- `run_metadata.json`

The preprocessing pipeline (imputation and one-hot encoding) is fitted only
within the training folds. Hyperparameter selection uses stratified
cross-validation and average precision. The classification threshold is
selected using out-of-fold predictions from the training set. The held-out
test set is not used for model selection or threshold tuning.

## Important methodological notes

1. **Target validity:** document the exact operational definition of
   `Y_deterioro`. A percentile-derived label is sample-relative and should not
   be described as a clinical diagnosis.
2. **Temporal prediction:** a true future-risk model requires a longitudinal
   design: predictors at an earlier wave and cognitive outcome at a later wave,
   with participant-level separation and a clearly defined prediction horizon.
3. **Repeated observations:** if multiple waves per participant are present,
   random row-wise splitting can leak participant-specific information.
   Consider participant-grouped or temporal validation.
4. **Survey design:** SHARE uses a complex sampling design. Discuss weights,
   strata, clusters, nonresponse, and cross-country comparability. Standard
   scikit-learn estimates do not automatically account for the survey design.
5. **Country:** country is excluded as a predictor by default to reduce the
   possibility that the model mainly learns geographic differences. Revisit
   this choice as a prespecified research question and report subgroup
   performance if relevant.
6. **Threshold:** the F1-optimal threshold is a statistical operating point,
   not a clinical decision threshold. Choose a clinically meaningful utility
   function before any real-world use.
7. **Interpretability:** permutation importance describes model reliance under
   this dataset and validation setup; it does not establish causality.
8. **External validation:** a held-out random test split is internal
   validation. Generalization requires temporal, geographic, or independent
   cohort validation.

## Data governance and SHARE access

SHARE data are subject to access, use, and redistribution conditions. Keep
the raw data, derived participant-level files, identifiers, and restricted
outputs out of the public repository unless the applicable SHARE terms and
your data-access agreement explicitly authorize their release. Publish code,
documentation, environment details, and synthetic/example data where allowed.
Check the current SHARE data access and citation requirements before release.

## Reproducibility checklist before release

- [ ] Verify the target formula and codebook mappings.
- [ ] Confirm no predictor directly contains SGC or its component items.
- [ ] Establish whether data contain one row per participant or repeated waves.
- [ ] Define the intended task: cross-sectional classification or longitudinal prediction.
- [ ] Decide how SHARE survey weights/design will be addressed.
- [ ] Add a data dictionary and provenance for every derived variable.
- [ ] Run the script from a clean virtual environment.
- [ ] Review generated metrics and subgroup performance.
- [ ] Add the exact SHARE citation and access statement required by your agreement.
- [ ] Confirm that no restricted data are committed to GitHub.

## License

No license is assigned yet. Choose a license for the code only after checking
the terms applicable to the SHARE data and any third-party code included.
A public GitHub repository without an explicit license is visible but does not
automatically grant reuse rights.
