"""WOE/IV credit-card fraud scorecard utilities.

The functions in this module deliberately fit bin boundaries on the training
sample only.  This avoids leaking information from the hold-out sample into
feature engineering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


DEFAULT_FEATURES = [
    "V14", "V12", "V3", "V4", "V11",
    "V10", "V17", "V2", "V16", "V27",
]


@dataclass(frozen=True)
class ScoreParameters:
    base_score: float = 400.0
    pdo: float = 35.0
    base_odds_good_to_bad: float = 100.0
    minimum: int = 0
    maximum: int = 1000


def load_creditcard_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate the Kaggle ULB credit-card dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for download instructions."
        )
    df = pd.read_csv(path)
    expected = {"Time", "Amount", "Class", *(f"V{i}" for i in range(1, 29))}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if not set(df["Class"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Class must contain only 0 (normal) and 1 (fraud).")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.30,
    random_state: int = 42,
):
    """Create a reproducible stratified train/test split."""
    X = df.drop(columns="Class")
    y = df["Class"].astype(int)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def _quantile_edges(series: pd.Series, n_bins: int) -> np.ndarray:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return np.array([-np.inf, np.inf])
    _, raw_edges = pd.qcut(clean, q=n_bins, retbins=True, duplicates="drop")
    edges = np.unique(raw_edges.astype(float))
    if len(edges) < 2:
        return np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def fit_woe_bins(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: Iterable[str],
    n_bins: int = 10,
    alpha: float = 0.5,
):
    """Fit quantile bins and Laplace-smoothed WOE mappings on training data."""
    definitions: dict[str, dict] = {}
    iv_rows = []
    y_train = y_train.reset_index(drop=True)

    for feature in features:
        values = X_train[feature].reset_index(drop=True)
        edges = _quantile_edges(values, n_bins)
        bucket = pd.cut(values, bins=edges, include_lowest=True, duplicates="drop")
        frame = pd.DataFrame({"bucket": bucket, "target": y_train})
        grouped = frame.groupby("bucket", observed=False)["target"].agg(["count", "sum"])
        grouped = grouped.rename(columns={"sum": "bad"})
        grouped["good"] = grouped["count"] - grouped["bad"]

        k = max(len(grouped), 1)
        dist_good = (grouped["good"] + alpha) / (grouped["good"].sum() + alpha * k)
        dist_bad = (grouped["bad"] + alpha) / (grouped["bad"].sum() + alpha * k)
        grouped["woe"] = np.log(dist_good / dist_bad)
        grouped["iv_component"] = (dist_good - dist_bad) * grouped["woe"]

        woe_values = grouped["woe"].astype(float).tolist()
        definitions[feature] = {
            "edges": edges.tolist(),
            "woe": woe_values,
            "missing_woe": 0.0,
        }
        iv_rows.append({"feature": feature, "iv": float(grouped["iv_component"].sum())})

    iv_table = pd.DataFrame(iv_rows).sort_values("iv", ascending=False).reset_index(drop=True)
    return definitions, iv_table


def transform_woe(X: pd.DataFrame, definitions: dict) -> pd.DataFrame:
    """Apply stored training-time bins and WOE values to any compatible sample."""
    transformed = pd.DataFrame(index=X.index)
    for feature, spec in definitions.items():
        edges = np.asarray(spec["edges"], dtype=float)
        codes = pd.cut(
            pd.to_numeric(X[feature], errors="coerce"),
            bins=edges,
            include_lowest=True,
            labels=False,
        )
        mapping = np.asarray(spec["woe"], dtype=float)
        values = np.full(len(X), float(spec.get("missing_woe", 0.0)))
        valid = codes.notna() & (codes >= 0) & (codes < len(mapping))
        values[valid.to_numpy()] = mapping[codes[valid].astype(int).to_numpy()]
        transformed[f"{feature}_WOE"] = values
    return transformed


def fit_scorecard(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: Iterable[str] = DEFAULT_FEATURES,
    n_bins: int = 10,
    alpha: float = 0.5,
    random_state: int = 42,
):
    """Fit WOE definitions and a class-weighted logistic-regression model."""
    features = list(features)
    definitions, iv_table = fit_woe_bins(
        X_train, y_train, features, n_bins=n_bins, alpha=alpha
    )
    X_train_woe = transform_woe(X_train, definitions)
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_train_woe, y_train)
    return model, definitions, iv_table


def ks_statistic(y_true, probability) -> float:
    """Return the maximum separation between fraud and normal score CDFs."""
    fpr, tpr, _ = roc_curve(y_true, probability)
    return float(np.max(tpr - fpr))


def evaluate(model, X_woe: pd.DataFrame, y_true: pd.Series) -> dict:
    probability = model.predict_proba(X_woe)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "ks": ks_statistic(y_true, probability),
        "probability": probability,
    }


def restore_population_probability(
    balanced_probability,
    population_bad_rate: float,
) -> np.ndarray:
    """Correct probabilities from a balanced-weight model to the real prior.

    ``class_weight='balanced'`` gives the two target classes equal total
    training weight, so its probability output is expressed under an effective
    bad-class prior of 0.5.  This prior-odds correction restores the observed
    population fraud rate without changing transaction ranking, ROC-AUC or KS.
    """
    if not 0 < population_bad_rate < 1:
        raise ValueError("population_bad_rate must be strictly between 0 and 1")
    p = np.clip(np.asarray(balanced_probability, dtype=float), 1e-12, 1 - 1e-12)
    balanced_odds = p / (1 - p)
    population_odds = population_bad_rate / (1 - population_bad_rate)
    corrected_odds = balanced_odds * population_odds
    return corrected_odds / (1 + corrected_odds)


def probability_to_score(
    probability,
    params: ScoreParameters | None = None,
) -> np.ndarray:
    """Convert fraud probability to a conventional 0–1000 score.

    A larger score means lower estimated fraud risk.  At the base score the
    good-to-bad odds equal ``base_odds_good_to_bad``; every PDO points doubles
    those odds.
    """
    params = params or ScoreParameters()
    p = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    good_to_bad_odds = (1 - p) / p
    factor = params.pdo / np.log(2)
    raw = params.base_score + factor * np.log(
        good_to_bad_odds / params.base_odds_good_to_bad
    )
    return np.rint(np.clip(raw, params.minimum, params.maximum)).astype(int)


def risk_decision(score) -> pd.DataFrame:
    """Map score to the agreed three-tier fraud-control strategy."""
    score = pd.Series(score)
    level = np.select(
        [score <= 300, score <= 500],
        ["High Risk", "Medium Risk"],
        default="Low Risk",
    )
    action = np.select(
        [score <= 300, score <= 500],
        ["Block Transaction", "Manual Review"],
        default="Approve",
    )
    return pd.DataFrame({"Risk_Level": level, "Recommended_Action": action}, index=score.index)


def build_bundle(
    model,
    definitions,
    features,
    score_parameters=None,
    population_bad_rate: float | None = None,
) -> dict:
    params = score_parameters or ScoreParameters()
    return {
        "model": model,
        "woe_definitions": definitions,
        "features": list(features),
        "score_parameters": params,
        "population_bad_rate": population_bad_rate,
        "version": "1.0",
    }


def save_bundle(bundle: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str | Path) -> dict:
    return joblib.load(path)


def score_transactions(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Return the original transactions with probability, score and action."""
    X_woe = transform_woe(df, bundle["woe_definitions"])
    raw_probability = bundle["model"].predict_proba(X_woe)[:, 1]
    population_bad_rate = bundle.get("population_bad_rate")
    probability = (
        restore_population_probability(raw_probability, population_bad_rate)
        if population_bad_rate is not None
        else raw_probability
    )
    score = probability_to_score(probability, bundle["score_parameters"])
    decisions = risk_decision(score)
    result = df.copy().reset_index(drop=True)
    result["Raw_Model_Probability"] = raw_probability
    result["Fraud_Probability"] = probability
    result["Risk_Score"] = score
    result = pd.concat([result, decisions.reset_index(drop=True)], axis=1)
    return result
