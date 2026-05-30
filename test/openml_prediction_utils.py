from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SMOOTH_MULTI_DIR = Path("/home/kerith/workspaces/smooth_multi")
MANIFEST_PATH = SMOOTH_MULTI_DIR / "manifest.csv"
TARGET_COLUMN = "__target__"
FRENCH_MOTOR_CLAIMS_SUITE = "OpenML-French-Motor-Claims"
FRENCH_MOTOR_CLAIMS_EXPOSURE_COLUMN = "Exposure"
ALLSTATE_CLAIMS_SEVERITY_SUITE = "OpenML-Allstate-Claims-Severity"
CLASSIFICATION_SUITES = {"OpenML-CC18", "OpenML-Covertype"}
REGRESSION_SUITES = {"OpenML-CTR23", FRENCH_MOTOR_CLAIMS_SUITE, ALLSTATE_CLAIMS_SEVERITY_SUITE}
DEFAULT_SUITES = {
    "OpenML-CC18",
    "OpenML-Covertype",
    "OpenML-CTR23",
    FRENCH_MOTOR_CLAIMS_SUITE,
    ALLSTATE_CLAIMS_SEVERITY_SUITE,
}
EXCLUDED_DATASETS = {"kr-vs-kp", "cnae-9"}
SPLIT_SEED = 42
DEPTHS = [1, 2, 3, 4, 5, 6]


@dataclass(frozen=True)
class DatasetTask:
    suite: str
    name: str
    csv_path: Path
    task_type: str


@dataclass(frozen=True)
class PreparedData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    weight_train: pd.Series | None
    weight_val: pd.Series | None
    cat_features: list[str]
    class_count: int | None
    class_labels: list[str]


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    value = re.sub(r"_+", "_", value).strip("_.")
    return value[:120] or "dataset"


def infer_task_type(suite_name: str) -> str:
    if suite_name in CLASSIFICATION_SUITES:
        return "classification"
    if suite_name in REGRESSION_SUITES:
        return "regression"
    raise ValueError(f"Unsupported suite: {suite_name}")


def load_tasks(manifest_path: Path = MANIFEST_PATH, suites: set[str] = DEFAULT_SUITES) -> list[DatasetTask]:
    manifest = pd.read_csv(manifest_path)
    tasks: list[DatasetTask] = []
    for _, row in manifest.iterrows():
        suite = str(row["suite"])
        if suite not in suites:
            continue
        dataset_name = str(row["dataset_name"])
        if dataset_name in EXCLUDED_DATASETS:
            continue
        tasks.append(
            DatasetTask(
                suite=suite,
                name=dataset_name,
                csv_path=Path(row["csv_path"]),
                task_type=infer_task_type(suite),
            )
        )
    return tasks


def get_cat_feature_names(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column])
        or pd.api.types.is_bool_dtype(frame[column])
    ]


def cast_integer_columns_to_float(frame: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if column in exclude:
            continue
        if pd.api.types.is_integer_dtype(result[column]):
            result[column] = result[column].astype(float)
    return result


def prepare_model_frame(frame: pd.DataFrame, suite_name: str) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    frame = frame.loc[frame[TARGET_COLUMN].notna()].reset_index(drop=True)
    if suite_name != FRENCH_MOTOR_CLAIMS_SUITE:
        return frame.drop(columns=[TARGET_COLUMN]), frame[TARGET_COLUMN], None

    if FRENCH_MOTOR_CLAIMS_EXPOSURE_COLUMN not in frame.columns:
        raise ValueError(
            f"{FRENCH_MOTOR_CLAIMS_SUITE} requires "
            f"{FRENCH_MOTOR_CLAIMS_EXPOSURE_COLUMN!r} for row weights."
        )

    exposure = pd.to_numeric(frame[FRENCH_MOTOR_CLAIMS_EXPOSURE_COLUMN], errors="coerce")
    claim_count = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    valid = claim_count.notna() & exposure.notna() & (exposure > 0)
    frame = frame.loc[valid].reset_index(drop=True)
    exposure = exposure.loc[valid].reset_index(drop=True).astype(float)
    claim_count = claim_count.loc[valid].reset_index(drop=True).astype(float)
    X = frame.drop(columns=[TARGET_COLUMN, FRENCH_MOTOR_CLAIMS_EXPOSURE_COLUMN])
    y = claim_count / exposure
    return X, y, exposure


def load_prepared_data(task: DatasetTask) -> PreparedData:
    frame = pd.read_csv(task.csv_path)
    X, y, row_weight = prepare_model_frame(frame, task.suite)
    X = cast_integer_columns_to_float(X, exclude=set())
    cat_features = get_cat_feature_names(X)
    for column in cat_features:
        X[column] = X[column].fillna("__nan__").astype(str)

    class_count = int(y.nunique(dropna=False)) if task.task_type == "classification" else None
    class_labels = [str(value) for value in sorted(y.dropna().unique())] if class_count else []

    stratify = y if task.task_type == "classification" and y.nunique(dropna=False) > 1 else None
    if row_weight is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=SPLIT_SEED,
            stratify=stratify,
        )
        weight_train = None
        weight_val = None
    else:
        split_values = train_test_split(
            X,
            y,
            row_weight,
            test_size=0.3,
            random_state=SPLIT_SEED,
            stratify=stratify,
        )
        X_train, X_val, y_train, y_val, weight_train, weight_val = split_values

    return PreparedData(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        weight_train=weight_train,
        weight_val=weight_val,
        cat_features=cat_features,
        class_count=class_count,
        class_labels=class_labels,
    )


def build_base_params(task: DatasetTask, prepared: PreparedData) -> tuple[type, dict]:
    params = {
        "allow_writing_files": False,
        "verbose": False,
        "thread_count": -1,
    }
    if task.task_type == "classification":
        loss_function = "Logloss" if (prepared.class_count or 0) <= 2 else "MultiClass"
        params["loss_function"] = loss_function
        params["eval_metric"] = loss_function
        return cb.CatBoostClassifier, params

    params["loss_function"] = "RMSE"
    params["eval_metric"] = "RMSE"
    return cb.CatBoostRegressor, params


def add_prediction_columns(predictions: pd.DataFrame, model, X_val: pd.DataFrame, seed: int, task: DatasetTask) -> None:
    if task.task_type == "regression":
        predictions[f"pred_seed_{seed}"] = model.predict(X_val)
        return

    probabilities = np.asarray(model.predict_proba(X_val))
    model_classes = [str(value) for value in model.classes_]
    if probabilities.shape[1] == 2:
        predictions[f"pred_seed_{seed}"] = probabilities[:, 1]
        predictions[f"pred_seed_{seed}_class"] = model_classes[1]
        return

    for class_index, class_label in enumerate(model_classes):
        predictions[f"pred_seed_{seed}_class_{safe_name(class_label)}"] = probabilities[:, class_index]
