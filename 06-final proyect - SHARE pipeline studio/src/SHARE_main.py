"""
SHARE cognitive-status classification
=====================================

Reproducible baseline workflow:
1. Load a locally prepared SHARE-derived analysis table.
2. Validate the target and exclude identifiers / direct target components.
3. Split before fitting preprocessing.
4. Tune a Random Forest using stratified CV and average precision.
5. Select a classification threshold from out-of-fold TRAIN predictions.
6. Evaluate once on the untouched TEST set.
7. Export metrics, predictions, plots, and the fitted pipeline.

IMPORTANT:
- This script does not establish a clinical diagnosis.
- If Y_deterioro is defined from the lower quartile of SGC, this is a
  cross-sectional classification of that operational label, not prediction
  of future cognitive decline.
- Review TARGET_DEFINITION and EXCLUDED_FEATURES against the SHARE codebook
  and the exact construction of your analytic dataset before publication.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance


# ---------------------------------------------------------------------
# 1. Configuration: edit this section for your analytic dataset
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "df_final_listo.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
N_ITER_SEARCH = 25
N_JOBS = -1

TARGET = "Y_deterioro"
COUNTRY_COLUMN = "country"

# Exclude direct target components and identifiers from predictors.
# Add other columns if they encode the target by construction.
EXCLUDED_FEATURES = {
    TARGET,
    "SGC",             # Direct source of Y_deterioro in the current project
    "deterioro",       # Potential duplicate/alternative target
    "mergeid",
    "mergeid_p",
    "id",
}

# Optional: exclude country as a predictor to reduce geographic proxy learning.
# Country may still be retained for descriptive subgroup analyses.
INCLUDE_COUNTRY_AS_PREDICTOR = False

# The current operational target described in the original script:
TARGET_DEFINITION = (
    "Y_deterioro: binary label derived from the lower quartile of SGC. "
    "Confirm the exact threshold, direction, and handling of ties from the "
    "dataset-generation code before publication."
)

# Optional columns that should be treated as categorical even if numeric.
CATEGORICAL_COLUMNS = [
    "sexo", "estado_civil", "ni", "tipo_ciudad", "imc_cat",
    "smoking", "acitividad_vigorosa",
    "depresion", "pesimismo", "pensamiento_suicida", "culplable",
    "problema_de_sueno", "irritable", "perdida_de_apetito",
    "fatiga", "problema_de_concentracion", "llorado",
]


# ---------------------------------------------------------------------
# 2. Logging and utilities
# ---------------------------------------------------------------------

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def make_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place your authorized analysis dataset at data/df_final_listo.csv "
            "or update DATA_PATH in src/main.py."
        )
    data = pd.read_csv(path, low_memory=False)
    if data.empty:
        raise ValueError("The input dataset is empty.")
    logging.info("Loaded dataset: %s rows × %s columns", *data.shape)
    return data


def validate_target(data: pd.DataFrame, target: str) -> pd.Series:
    if target not in data.columns:
        raise KeyError(f"Target column '{target}' is absent.")
    y = data[target]
    if y.isna().any():
        raise ValueError(
            f"Target '{target}' contains {int(y.isna().sum())} missing values. "
            "Resolve the target definition before modeling; do not impute labels."
        )
    observed = set(pd.Series(y).unique())
    if not observed.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(
            f"Expected a binary 0/1 target; observed values: {sorted(observed, key=str)}"
        )
    y = y.astype("int8")
    if y.nunique() != 2:
        raise ValueError("Both target classes must be present.")
    return y


def summarize_data(data: pd.DataFrame, y: pd.Series) -> None:
    missing = (
        data.isna().mean()
        .mul(100)
        .sort_values(ascending=False)
        .rename("missing_percent")
        .to_frame()
    )
    missing.to_csv(OUTPUT_DIR / "missingness_summary.csv")
    class_summary = (
        y.value_counts(dropna=False)
        .sort_index()
        .rename_axis("class")
        .to_frame("n")
    )
    class_summary["percent"] = class_summary["n"] / len(y) * 100
    class_summary.to_csv(OUTPUT_DIR / "target_distribution.csv")
    logging.info("Target distribution:\n%s", class_summary)


def build_feature_table(data: pd.DataFrame) -> pd.DataFrame:
    excluded = set(EXCLUDED_FEATURES)
    if not INCLUDE_COUNTRY_AS_PREDICTOR:
        excluded.add(COUNTRY_COLUMN)

    features = data.drop(
        columns=[c for c in excluded if c in data.columns],
        errors="ignore",
    ).copy()

    # Remove columns that are entirely missing or constant.
    all_missing = features.columns[features.isna().all()].tolist()
    constant = [
        c for c in features.columns
        if c not in all_missing and features[c].nunique(dropna=True) <= 1
    ]
    remove = sorted(set(all_missing + constant))
    if remove:
        logging.info("Dropping all-missing/constant features: %s", remove)
        features = features.drop(columns=remove)

    if features.shape[1] == 0:
        raise ValueError("No predictor columns remain after exclusions.")

    # Ensure columns with mixed types are handled consistently.
    for col in features.select_dtypes(include=["object", "string"]).columns:
        features[col] = features[col].astype("string")

    return features


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical = [
        c for c in X.columns
        if c in CATEGORICAL_COLUMNS
        or pd.api.types.is_object_dtype(X[c])
        or pd.api.types.is_string_dtype(X[c])
        or isinstance(X[c].dtype, pd.CategoricalDtype)
    ]
    numeric = [c for c in X.columns if c not in categorical]

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", classifier),
        ]
    )


def choose_threshold_from_oof(y_train: pd.Series, oof_probabilities: np.ndarray) -> dict[str, float]:
    """Select threshold on TRAIN out-of-fold probabilities, never on TEST."""
    precision, recall, thresholds = precision_recall_curve(
        y_train, oof_probabilities
    )
    if len(thresholds) == 0:
        return {"threshold": 0.5, "f1_oof": float("nan")}

    # precision and recall have one extra endpoint relative to thresholds.
    p = precision[:-1]
    r = recall[:-1]
    f1 = np.divide(
        2 * p * r,
        p + r,
        out=np.zeros_like(p),
        where=(p + r) > 0,
    )
    best_idx = int(np.nanargmax(f1))
    return {
        "threshold": float(thresholds[best_idx]),
        "f1_oof": float(f1[best_idx]),
    }


def save_evaluation(
    y_test: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    test_index: pd.Index,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype("int8")
    metrics = {
        "n_test": int(len(y_test)),
        "prevalence_test": float(y_test.mean()),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "threshold_selected_on_train_oof": float(threshold),
        "f1_at_selected_threshold": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Y=0", "Y=1"],
            output_dict=True,
            zero_division=0,
        ),
    }

    pd.DataFrame(
        {
            "observed": y_test.to_numpy(),
            "predicted_probability": probabilities,
            "predicted_class": predictions,
        },
        index=test_index,
    ).to_csv(OUTPUT_DIR / "test_predictions.csv", index_label="row_index")

    with (OUTPUT_DIR / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logging.info(
        "TEST results | AP=%.3f | ROC-AUC=%.3f | F1=%.3f | threshold=%.3f",
        metrics["average_precision"],
        metrics["roc_auc"],
        metrics["f1_at_selected_threshold"],
        threshold,
    )
    return metrics


def save_plots(y_test: pd.Series, probabilities: np.ndarray, threshold: float) -> None:
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=axes[0])
    axes[0].set_title("ROC curve — held-out test set")

    PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=axes[1])
    axes[1].axhline(
        y=float(y_test.mean()),
        linestyle="--",
        color="grey",
        label=f"Test prevalence = {y_test.mean():.3f}",
    )
    axes[1].axvline(threshold, linestyle=":", color="black", label=f"Threshold = {threshold:.3f}")
    axes[1].set_title("Precision–recall curve — held-out test set")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_performance_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_logging()
    make_output_dirs()

    data = load_data(DATA_PATH)
    y = validate_target(data, TARGET)
    summarize_data(data, y)

    X = build_feature_table(data)
    if len(X) != len(y):
        raise RuntimeError("Predictor and target row counts do not match.")

    # Split BEFORE fitting imputers, encoders, feature transforms, or models.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    logging.info(
        "Train: %s rows | Test: %s rows | Features: %s",
        len(X_train), len(X_test), X.shape[1],
    )

    pipeline = build_pipeline(X_train)
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    param_distributions = {
        "classifier__n_estimators": [300, 500, 800],
        "classifier__max_depth": [None, 5, 8, 12, 16],
        "classifier__min_samples_split": [5, 10, 20],
        "classifier__min_samples_leaf": [2, 5, 10, 15],
        "classifier__max_features": ["sqrt", 0.3, 0.5, 0.8],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=N_ITER_SEARCH,
        scoring="average_precision",
        cv=cv,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    cv_results = pd.DataFrame(search.cv_results_).sort_values(
        "rank_test_score"
    )
    cv_results.to_csv(OUTPUT_DIR / "hyperparameter_search.csv", index=False)
    logging.info("Best CV average precision: %.4f", search.best_score_)
    logging.info("Best parameters: %s", search.best_params_)

    # Obtain out-of-fold probabilities using TRAIN only for threshold selection.
    oof_probabilities = cross_val_predict(
        best_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=N_JOBS,
    )[:, 1]
    threshold_result = choose_threshold_from_oof(y_train, oof_probabilities)
    selected_threshold = threshold_result["threshold"]
    logging.info(
        "Threshold selected from TRAIN OOF predictions: %.4f (OOF F1=%.4f)",
        selected_threshold,
        threshold_result["f1_oof"],
    )

    # Refit on all training data (already done by RandomizedSearchCV refit=True).
    # Evaluate exactly once on the untouched test set.
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    metrics = save_evaluation(
        y_test, test_probabilities, selected_threshold, X_test.index
    )
    save_plots(y_test, test_probabilities, selected_threshold)

    # Permutation importance is reported on held-out test data as descriptive
    # model reliance, not as causal effect or independent clinical importance.
    perm = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(OUTPUT_DIR / "permutation_importance.csv", index=False)

    joblib.dump(best_model, OUTPUT_DIR / "random_forest_pipeline.joblib")

    metadata = {
        "target": TARGET,
        "target_definition": TARGET_DEFINITION,
        "excluded_features": sorted(EXCLUDED_FEATURES),
        "include_country_as_predictor": INCLUDE_COUNTRY_AS_PREDICTOR,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_splits": N_SPLITS,
        "best_cv_average_precision": float(search.best_score_),
        "best_parameters": search.best_params_,
        "threshold_selection": "max F1 on out-of-fold TRAIN probabilities",
        "test_metrics_file": "test_metrics.json",
        "n_rows": int(len(data)),
        "n_features_before_encoding": int(X.shape[1]),
    }
    with (OUTPUT_DIR / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Analysis complete. Results saved to: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
