"""Machine-learning analysis for radiomics-based differential diagnosis.

The positive class is radicular cyst. Levene-based Student/Welch t-test
feature screening, feature scaling, SMOTE, and hyperparameter tuning are performed
inside training folds.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False

try:
    from .delong import paired_delong_test
except ImportError:  # pragma: no cover
    from delong import paired_delong_test


class LeveneTTestFeatureSelector(BaseEstimator):
    """Manuscript-matched univariate feature screening.

    For each feature, Levene's test is first applied to the two diagnostic
    groups. If the group variances are not significantly different, a Student
    two-sample t-test is used. Otherwise, Welch's t-test is used. Features with
    t-test p-values below ``p_threshold`` are retained.

    This selector must be fitted only on the training data within each CV
    fold to avoid information leakage.
    """

    def __init__(self, levene_alpha: float = 0.05, p_threshold: float = 0.05):
        self.levene_alpha = levene_alpha
        self.p_threshold = p_threshold

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y).astype(int)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D feature matrix.")
        if np.isnan(X_arr).any():
            raise ValueError("Missing values detected before feature screening.")

        self.n_features_in_ = X_arr.shape[1]
        self.p_values_ = np.full(self.n_features_in_, np.nan, dtype=float)
        self.levene_p_values_ = np.full(self.n_features_in_, np.nan, dtype=float)
        self.equal_var_ = np.zeros(self.n_features_in_, dtype=bool)
        self.test_used_ = np.array(["not_tested"] * self.n_features_in_, dtype=object)

        group0 = X_arr[y_arr == 0]
        group1 = X_arr[y_arr == 1]
        if group0.shape[0] < 2 or group1.shape[0] < 2:
            raise ValueError("Both classes must contain at least two samples for t-test screening.")

        for j in range(self.n_features_in_):
            x0 = group0[:, j]
            x1 = group1[:, j]
            if np.nanstd(x0) == 0 and np.nanstd(x1) == 0:
                continue
            try:
                levene_p = stats.levene(x0, x1, center="median").pvalue
            except Exception:
                levene_p = np.nan
            equal_var = bool(np.isfinite(levene_p) and levene_p >= self.levene_alpha)
            try:
                ttest = stats.ttest_ind(x0, x1, equal_var=equal_var, nan_policy="raise")
                p_val = float(ttest.pvalue)
            except Exception:
                p_val = np.nan
            self.levene_p_values_[j] = levene_p
            self.equal_var_[j] = equal_var
            self.test_used_[j] = "student_t_test" if equal_var else "welch_t_test"
            self.p_values_[j] = p_val

        self.support_mask_ = np.isfinite(self.p_values_) & (self.p_values_ < self.p_threshold)
        if not np.any(self.support_mask_):
            raise ValueError(
                "No features passed Levene-based Student/Welch t-test screening "
                f"at p < {self.p_threshold}."
            )
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return X_arr[:, self.support_mask_]

    def get_support(self, indices=False):
        if indices:
            return np.where(self.support_mask_)[0]
        return self.support_mask_

    def screening_table(self, feature_names: list[str] | None = None) -> pd.DataFrame:
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.n_features_in_)]
        return pd.DataFrame({
            "feature": list(feature_names),
            "levene_p_value": self.levene_p_values_,
            "equal_variance_assumed": self.equal_var_,
            "test_used": self.test_used_,
            "ttest_p_value": self.p_values_,
            "retained": self.support_mask_,
        })


def load_feature_table(path: str | Path, index_col: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=index_col)
    df.index = df.index.astype(str)
    return df.sort_index()


def make_dd_labels(df: pd.DataFrame, label_column: str | None = None, cyst_suffix: str = "C") -> pd.Series:
    if label_column is not None:
        y = df[label_column].copy()
        if y.dtype == object:
            y = y.astype(str).str.strip().str.lower().map({
                "granuloma": 0, "g": 0, "0": 0,
                "cyst": 1, "c": 1, "1": 1,
            })
        y = y.astype(int)
    else:
        y = df.index.to_series().str.endswith(cyst_suffix).astype(int)
    return y.rename("y")


def select_radiomic_columns(df: pd.DataFrame, label_column: str | None = None) -> list[str]:
    excluded = set()
    if label_column is not None:
        excluded.add(label_column)
    cols = [c for c in df.columns if c not in excluded and str(c).startswith("original_")]
    if not cols:
        raise ValueError("No PyRadiomics columns starting with 'original_' were found.")
    return cols


def prepare_feature_matrices(
    raw_csv: str | Path,
    normalized_csv: str | Path,
    label_column: str | None = None,
    cyst_suffix: str = "C",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw_df = load_feature_table(raw_csv)
    norm_df = load_feature_table(normalized_csv)
    common_cases = raw_df.index.intersection(norm_df.index)

    raw_df = raw_df.loc[common_cases].copy()
    norm_df = norm_df.loc[common_cases].copy()
    y = make_dd_labels(norm_df, label_column=label_column, cyst_suffix=cyst_suffix)

    raw_cols = select_radiomic_columns(raw_df, label_column=label_column)
    norm_cols = select_radiomic_columns(norm_df, label_column=label_column)
    common_features = sorted(set(raw_cols).intersection(norm_cols))

    X_raw = raw_df[common_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_norm = norm_df[common_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    valid = y.notna()
    X_raw = X_raw.loc[valid]
    X_norm = X_norm.loc[valid]
    y = y.loc[valid].astype(int)

    print(f"Number of cases: {len(y)}")
    print(f"Class distribution: {Counter(y)}")
    print(f"Common radiomic features: {len(common_features)}")
    return X_raw, X_norm, y


def build_feature_sets(columns: list[str]) -> dict[str, list[str]]:
    columns = list(columns)
    size_features = [c for c in columns if c.startswith("original_shape")]
    non_size_features = [c for c in columns if not c.startswith("original_shape")]
    return {
        name: feats
        for name, feats in {
            "size_only": size_features,
            "non_size_radiomics": non_size_features,
            "full_radiomics": columns,
        }.items()
        if len(feats) > 0
    }


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "AUC": float(roc_auc_score(y_true, y_prob)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall_Sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def bootstrap_auc_ci(y_true, y_prob, n_bootstrap: int = 2000, seed: int = 2048, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) == 2:
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not aucs:
        return {"AUC_CI_lower": np.nan, "AUC_CI_upper": np.nan}
    return {
        "AUC_CI_lower": float(np.percentile(aucs, 100 * alpha / 2)),
        "AUC_CI_upper": float(np.percentile(aucs, 100 * (1 - alpha / 2))),
    }


def safe_smote_k_options(y_train, inner_splits: int) -> list[int]:
    counts = pd.Series(y_train).value_counts()
    minority_count = int(counts.min())
    min_inner_minority = int(np.floor(minority_count * (inner_splits - 1) / inner_splits))
    max_k = min(5, min_inner_minority - 1)
    return [k for k in [1, 2, 3, 5] if k <= max_k]


def effective_n_iter(param_space: dict, requested: int) -> int:
    total = 1
    for values in param_space.values():
        try:
            total *= len(values)
        except TypeError:
            return requested
    return min(requested, total)


def get_model_and_param_space(
    model_name: str,
    n_features: int,
    y_train,
    use_smote: bool,
    random_state: int,
    inner_splits: int,
):
    class_weight_options = [None] if use_smote else [None, "balanced"]

    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            max_iter=5000,
            random_state=random_state,
        )
        param_space = {
            "clf__C": np.logspace(-3, 2, 20),
            "clf__l1_ratio": np.linspace(0.0, 1.0, 6),
            "clf__class_weight": class_weight_options,
        }
    elif model_name == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed.")
        y_arr = np.asarray(y_train)
        pos = int(np.sum(y_arr == 1))
        neg = int(np.sum(y_arr == 0))
        scale_pos_weight = 1.0 if use_smote else neg / max(pos, 1)
        classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=random_state,
            seed=random_state,
            n_jobs=-1,
        )
        param_space = {
            "clf__n_estimators": [50, 100, 200, 300],
            "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "clf__max_depth": [2, 3, 4],
            "clf__min_child_weight": [1, 3, 5],
            "clf__subsample": [0.7, 0.85, 1.0],
            "clf__colsample_bytree": [0.7, 0.85, 1.0],
            "clf__gamma": [0, 0.1, 0.3],
            "clf__scale_pos_weight": [scale_pos_weight],
        }
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    steps = [
        ("select", LeveneTTestFeatureSelector(levene_alpha=0.05, p_threshold=0.05)),
        ("scale", StandardScaler()),
    ]

    if use_smote:
        smote_k = safe_smote_k_options(y_train, inner_splits)
        if len(smote_k) == 0:
            raise ValueError("Too few minority-class samples for SMOTE inside inner CV.")
        steps.append(("smote", SMOTE(random_state=random_state)))
        param_space["smote__k_neighbors"] = smote_k
        param_space["smote__sampling_strategy"] = ["auto"]

    steps.append(("clf", classifier))
    pipeline = ImbPipeline(steps) if use_smote else SklearnPipeline(steps)
    return pipeline, param_space


def selected_feature_names(fitted_pipeline, feature_names: list[str]) -> list[str]:
    names = pd.Index(feature_names)
    support = fitted_pipeline.named_steps["select"].get_support()
    return list(names[support])


def model_importance(fitted_pipeline, feature_names: list[str]) -> pd.DataFrame:
    selected = selected_feature_names(fitted_pipeline, feature_names)
    clf = fitted_pipeline.named_steps["clf"]
    if hasattr(clf, "coef_"):
        values = np.abs(clf.coef_).ravel()
        importance_type = "absolute_standardized_coefficient"
    elif hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
        importance_type = "model_native_feature_importance"
    else:
        return pd.DataFrame()
    return pd.DataFrame({"feature": selected, "importance": values, "importance_type": importance_type})


def run_nested_cv_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
    feature_set_name: str,
    model_name: str,
    n_outer_splits: int = 5,
    n_inner_splits: int = 3,
    n_random_search_iter: int = 30,
    scoring: str = "roc_auc",
    use_smote: bool = True,
    random_state: int = 2048,
    search_n_jobs: int = 1,
) -> dict[str, pd.DataFrame]:
    X = X.copy()
    y = pd.Series(y, index=X.index).astype(int)
    if X.isna().any().any():
        missing_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "Missing values detected in the feature matrix. "
            "The manuscript workflow did not apply imputation. "
            f"Please review or remove missing values before model training. Columns: {missing_cols}"
        )

    outer_cv = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=random_state)
    fold_rows, pred_rows, param_rows, importance_rows, screening_rows = [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline, param_space = get_model_and_param_space(
            model_name=model_name,
            n_features=X_train.shape[1],
            y_train=y_train,
            use_smote=use_smote,
            random_state=random_state + fold,
            inner_splits=n_inner_splits,
        )
        inner_cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state + fold)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_space,
            n_iter=effective_n_iter(param_space, n_random_search_iter),
            scoring=scoring,
            cv=inner_cv,
            random_state=random_state + fold,
            n_jobs=search_n_jobs,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        y_prob = best_model.predict_proba(X_test)[:, 1]

        screening = best_model.named_steps["select"].screening_table(list(X.columns))
        screening["dataset"] = dataset_name
        screening["feature_set"] = feature_set_name
        screening["model"] = model_name
        screening["fold"] = fold
        screening_rows.append(screening)

        metrics = classification_metrics(y_test, y_prob)
        metrics.update({
            "dataset": dataset_name,
            "feature_set": feature_set_name,
            "model": model_name,
            "fold": fold,
            "inner_best_auc": float(search.best_score_),
            "n_selected_features": len(selected_feature_names(best_model, list(X.columns))),
        })
        fold_rows.append(metrics)

        pred_rows.append(pd.DataFrame({
            "case_id": X_test.index,
            "y_true": y_test.values,
            "y_prob": y_prob,
            "dataset": dataset_name,
            "feature_set": feature_set_name,
            "model": model_name,
            "fold": fold,
        }))

        param_rows.append({
            "dataset": dataset_name,
            "feature_set": feature_set_name,
            "model": model_name,
            "fold": fold,
            "best_inner_auc": float(search.best_score_),
            "best_params": json.dumps(search.best_params_),
        })

        imp = model_importance(best_model, list(X.columns))
        if not imp.empty:
            imp["dataset"] = dataset_name
            imp["feature_set"] = feature_set_name
            imp["model"] = model_name
            imp["fold"] = fold
            importance_rows.append(imp)

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(pred_rows, ignore_index=True)
    best_params = pd.DataFrame(param_rows)
    importances = pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame()
    feature_screening = pd.concat(screening_rows, ignore_index=True) if screening_rows else pd.DataFrame()

    pooled = classification_metrics(predictions["y_true"], predictions["y_prob"])
    pooled.update(bootstrap_auc_ci(predictions["y_true"], predictions["y_prob"], seed=random_state))
    pooled.update({
        "dataset": dataset_name,
        "feature_set": feature_set_name,
        "model": model_name,
        "n_cases": int(len(predictions)),
        "n_features": int(X.shape[1]),
    })

    return {
        "fold_metrics": fold_metrics,
        "predictions": predictions,
        "pooled_metrics": pd.DataFrame([pooled]),
        "best_params": best_params,
        "feature_screening_by_fold": feature_screening,
        "importances": importances,
    }


def compare_raw_vs_normalized(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature_set, model), group in predictions_df.groupby(["feature_set", "model"]):
        raw = group[group["dataset"] == "raw"]
        norm = group[group["dataset"] == "dentin_normalized"]
        if raw.empty or norm.empty:
            continue
        merged = raw[["case_id", "y_true", "y_prob"]].merge(
            norm[["case_id", "y_true", "y_prob"]],
            on="case_id",
            suffixes=("_raw", "_norm"),
        )
        if not np.array_equal(merged["y_true_raw"].values, merged["y_true_norm"].values):
            raise ValueError("Raw and normalized prediction labels are mismatched.")
        test = paired_delong_test(
            merged["y_true_raw"].values,
            merged["y_prob_raw"].values,
            merged["y_prob_norm"].values,
        )
        rows.append({
            "feature_set": feature_set,
            "model": model,
            "AUC_raw": test["auc_model_1"],
            "AUC_dentin_normalized": test["auc_model_2"],
            "AUC_difference_raw_minus_normalized": test["auc_difference_model_1_minus_model_2"],
            "p_value": test["p_value"],
            "n_cases": len(merged),
        })
    return pd.DataFrame(rows)


def summarize_importance(importance_df: pd.DataFrame) -> pd.DataFrame:
    if importance_df.empty:
        return pd.DataFrame()
    return (
        importance_df
        .groupby(["dataset", "feature_set", "model", "feature"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            selected_folds=("fold", "nunique"),
            importance_type=("importance_type", "first"),
        )
        .sort_values(["dataset", "feature_set", "model", "mean_importance"], ascending=[True, True, True, False])
    )


def run_all_experiments(
    raw_csv: str | Path,
    normalized_csv: str | Path,
    output_dir: str | Path,
    label_column: str | None = None,
    cyst_suffix: str = "C",
    model_names: list[str] | None = None,
    random_state: int = 2048,
    use_smote: bool = True,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_raw, X_norm, y = prepare_feature_matrices(raw_csv, normalized_csv, label_column, cyst_suffix)
    feature_sets = build_feature_sets(list(X_norm.columns))

    if model_names is None:
        model_names = ["logistic_regression"]
        if XGBOOST_AVAILABLE:
            model_names.append("xgboost")

    results = []
    for dataset_name, X_dataset in {"raw": X_raw, "dentin_normalized": X_norm}.items():
        for feature_set_name, cols in feature_sets.items():
            for model_name in model_names:
                print(f"Running {dataset_name} | {feature_set_name} | {model_name}")
                results.append(run_nested_cv_experiment(
                    X_dataset[cols], y,
                    dataset_name=dataset_name,
                    feature_set_name=feature_set_name,
                    model_name=model_name,
                    use_smote=use_smote,
                    random_state=random_state,
                    search_n_jobs=1,
                ))

    fold_metrics = pd.concat([r["fold_metrics"] for r in results], ignore_index=True)
    predictions = pd.concat([r["predictions"] for r in results], ignore_index=True)
    pooled_metrics = pd.concat([r["pooled_metrics"] for r in results], ignore_index=True)
    best_params = pd.concat([r["best_params"] for r in results], ignore_index=True)
    importance = pd.concat([r["importances"] for r in results if not r["importances"].empty], ignore_index=True)
    feature_screening = pd.concat([r["feature_screening_by_fold"] for r in results if not r["feature_screening_by_fold"].empty], ignore_index=True)
    raw_vs_norm = compare_raw_vs_normalized(predictions)
    importance_summary = summarize_importance(importance)

    outputs = {
        "fold_metrics": fold_metrics,
        "outer_fold_predictions": predictions,
        "pooled_metrics": pooled_metrics,
        "best_hyperparameters_by_fold": best_params,
        "feature_screening_by_fold": feature_screening,
        "raw_vs_dentin_normalized_delong": raw_vs_norm,
        "foldwise_feature_importance": importance,
        "foldwise_feature_importance_summary": importance_summary,
    }
    for name, df in outputs.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
    return outputs
