"""Общие утилиты: загрузка данных, препроцессинг, метрики."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLS = ["IC50, mM", "CC50, mM", "SI"]
DATA_PATH = "data/compounds.xlsx"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")
    return df


def get_features(df: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    drop = list(TARGET_COLS)
    if exclude:
        drop += exclude
    X = df.drop(columns=drop, errors="ignore")
    return X.select_dtypes(include=["number"])


def make_pipeline(model) -> Pipeline:
    """Стандартный пайплайн: импьютер → фильтр дисперсии → стандартизация → модель."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def log_transform_target(series: pd.Series) -> pd.Series:
    return np.log1p(series)
