import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


def load_config(config_path: str) -> Dict:
    """Load YAML configuration."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        config = yaml.safe_load(f)
    return config


def load_data(csv_path: str) -> pd.DataFrame:
    """Load full CSV without filtering."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Loaded CSV is empty.")
    return df


def _validate_columns(df: pd.DataFrame, target_col: str, group_col: str, feature_exclude: list) -> None:
    """Ensure required columns exist and no feature columns are missing."""
    missing = []
    for col in [target_col, group_col]:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    # Features are all columns except those explicitly excluded
    feature_cols = [c for c in df.columns if c not in feature_exclude]
    if not feature_cols:
        raise ValueError("No feature columns left after exclusions.")


def prepare_all_columns(
    df: pd.DataFrame,
    target_col: str,
    exclude_feature_cols: list,
) -> Tuple[np.ndarray, StandardScaler, pd.DataFrame, list]:
    """
    Minimal technical encoding and scaling for all feature columns.

    Returns:
        X_all_scaled: ndarray of all rows/features
        scaler: fitted StandardScaler
        df_with_clusters_base: copy of df to which clusters will be attached
        feature_cols: list of feature column names
    """
    feature_cols = [c for c in df.columns if c not in exclude_feature_cols]
    X_all = df[feature_cols].copy()

    for col in X_all.columns:
        series = X_all[col]

        # Datetime columns: detect by name containing "date"
        if "date" in col.lower():
            dt = pd.to_datetime(series, errors="coerce", utc=True)
            # Convert to int64 (ns since epoch); NaT becomes NaN
            X_all[col] = dt.astype("int64")
            X_all[col] = X_all[col].replace({np.iinfo(np.int64).min: np.nan})
            X_all[col] = X_all[col].fillna(0)

        # Numeric columns kept as-is (with median imputation)
        elif pd.api.types.is_numeric_dtype(series):
            X_all[col] = series.astype(float)
            X_all[col] = X_all[col].fillna(X_all[col].median())

        # Boolean columns: simple 0/1
        elif pd.api.types.is_bool_dtype(series):
            X_all[col] = series.astype(int)

        # Everything else (object, string, category): minimal ordinal encoding
        else:
            X_all[col] = series.astype(object).where(series.notna(), "__MISSING__")
            codes, _ = pd.factorize(X_all[col], sort=True)
            X_all[col] = codes.astype(float)

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all.values)

    df_with_clusters_base = df.copy()
    return X_all_scaled, scaler, df_with_clusters_base, feature_cols


def split_supervised_by_group(
    df_with_clusters: pd.DataFrame,
    target_col: str,
    group_col: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build supervised subset (target and group not null) and split by group.
    """
    mask_sup = df_with_clusters[target_col].notna() & df_with_clusters[group_col].notna()
    sup = df_with_clusters.loc[mask_sup].copy()

    if sup.empty:
        raise ValueError("No supervised rows: target or group column is entirely missing.")

    # Robust conversion of target to 0/1
    y_str = sup[target_col].astype(str).str.lower().str.strip()
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0}
    if set(y_str.unique()) - set(mapping.keys()):
        raise ValueError(
            f"Unexpected target values for '{target_col}'. Expected subset of {list(mapping.keys())}, "
            f"got: {sorted(set(y_str.unique()))}"
        )
    sup[target_col] = y_str.map(mapping).astype(int)

    groups = sup[group_col]

    if groups.nunique() < 2:
        raise ValueError(
            f"Need at least 2 distinct groups in '{group_col}' for GroupShuffleSplit. "
            f"Got {groups.nunique()}."
        )

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, val_idx = next(splitter.split(sup, sup[target_col], groups=groups))

    sup_train = sup.iloc[train_idx].copy()
    sup_val = sup.iloc[val_idx].copy()

    return sup_train, sup_val


def fit_kmeans_model(
    X_all_scaled: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> Tuple[KMeans, np.ndarray]:
    """Fit KMeans on all rows and return model and cluster labels."""
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    all_clusters = kmeans.fit_predict(X_all_scaled)
    return kmeans, all_clusters


def evaluate_predictions(
    sup_train: pd.DataFrame,
    sup_val: pd.DataFrame,
    target_col: str,
    threshold: float,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Compute cluster terminal rates, predictions on validation, and metrics.
    """
    if "cluster_id" not in sup_train.columns or "cluster_id" not in sup_val.columns:
        raise ValueError("cluster_id column must be present in supervised train/val data.")

    # Cluster terminal rates on train
    agg = sup_train.groupby("cluster_id")[target_col].agg(["count", "sum", "mean"])
    agg = agg.rename(
        columns={
            "count": "n_rows",
            "sum": "n_positives",
            "mean": "terminal_rate",
        }
    )
    agg = agg.reset_index()
    cluster_rates = agg.set_index("cluster_id")["terminal_rate"]
    global_rate = sup_train[target_col].mean()

    # Predictions on validation
    sup_val = sup_val.copy()
    sup_val["pred_proba"] = sup_val["cluster_id"].map(cluster_rates).fillna(global_rate)
    sup_val["pred_class"] = (sup_val["pred_proba"] >= threshold).astype(int)

    y_true = sup_val[target_col].values
    y_score = sup_val["pred_proba"].values
    y_pred = sup_val["pred_class"].values

    metrics: Dict[str, float] = {}

    # Some metrics require both classes to be present
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["roc_auc"] = None

    try:
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        metrics["average_precision"] = None

    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["confusion_matrix"] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "matrix": cm.tolist(),
    }

    return metrics, agg, sup_val


def save_outputs(
    df_with_clusters: pd.DataFrame,
    cluster_rates_df: pd.DataFrame,
    sup_val_with_preds: pd.DataFrame,
    metrics: Dict,
    scaler: StandardScaler,
    kmeans: KMeans,
    output_dir: str,
) -> None:
    """Persist metrics, CSVs, and models."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # All rows with clusters
    df_with_clusters.to_csv(out_path / "all_rows_with_clusters.csv", index=False)

    # Cluster terminal rates
    cluster_rates_df.to_csv(out_path / "cluster_terminal_rates.csv", index=False)

    # Validation predictions only
    sup_val_with_preds.to_csv(out_path / "val_predictions.csv", index=False)

    # Metrics
    with (out_path / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    # Models
    joblib.dump(scaler, out_path / "scaler.joblib")
    joblib.dump(kmeans, out_path / "kmeans.joblib")


def _print_summary(
    df: pd.DataFrame,
    sup_train: pd.DataFrame,
    sup_val: pd.DataFrame,
    target_col: str,
) -> None:
    """Print simple text summary."""
    n_total = len(df)
    n_sup = len(sup_train) + len(sup_val)

    train_pos_rate = sup_train[target_col].mean() if not sup_train.empty else float("nan")
    val_pos_rate = sup_val[target_col].mean() if not sup_val.empty else float("nan")

    print("=== Dataset summary ===")
    print(f"Total rows: {n_total}")
    print(f"Supervised rows (target & group known): {n_sup}")
    print(f"Train supervised size: {len(sup_train)}")
    print(f"Validation supervised size: {len(sup_val)}")
    print(f"Train positive rate: {train_pos_rate:.4f}")
    print(f"Validation positive rate: {val_pos_rate:.4f}")

    print("\nCluster distribution on all rows:")
    print(df["cluster_id"].value_counts(normalize=True).sort_index())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Terminal lightning clustering model with KMeans baseline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    csv_path = config["data"]["csv_path"]
    target_col = config["data"]["target_col"]
    group_col = config["data"]["group_col"]
    exclude_feature_cols = config["data"].get("exclude_feature_cols", [])

    n_clusters = int(config["model"]["n_clusters"])
    threshold = float(config["model"]["threshold"])
    random_state = int(config["model"]["random_state"])
    fit_on_all_rows = bool(config["model"].get("fit_on_all_rows", True))

    test_size = float(config["split"]["test_size"])
    output_dir = config["output"]["output_dir"]

    df = load_data(csv_path)

    _validate_columns(df, target_col, group_col, exclude_feature_cols)

    # Prepare all features and scale
    X_all_scaled, scaler, df_with_clusters, feature_cols = prepare_all_columns(
        df, target_col, exclude_feature_cols
    )

    if not fit_on_all_rows:
        # Option exists for future use; V0 fits on all rows by design.
        pass

    # Fit KMeans on all rows
    kmeans, all_clusters = fit_kmeans_model(
        X_all_scaled, n_clusters=n_clusters, random_state=random_state
    )
    df_with_clusters["cluster_id"] = all_clusters

    # Supervised split and evaluation
    sup_train, sup_val = split_supervised_by_group(
        df_with_clusters=df_with_clusters,
        target_col=target_col,
        group_col=group_col,
        test_size=test_size,
        random_state=random_state,
    )

    metrics, cluster_rates_df, sup_val_with_preds = evaluate_predictions(
        sup_train=sup_train,
        sup_val=sup_val,
        target_col=target_col,
        threshold=threshold,
    )

    save_outputs(
        df_with_clusters=df_with_clusters,
        cluster_rates_df=cluster_rates_df,
        sup_val_with_preds=sup_val_with_preds,
        metrics=metrics,
        scaler=scaler,
        kmeans=kmeans,
        output_dir=output_dir,
    )

    _print_summary(df_with_clusters, sup_train, sup_val, target_col)


if __name__ == "__main__":
    main()

