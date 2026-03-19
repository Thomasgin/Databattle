import pathlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RANDOM_STATE = 42


def build_preprocessor(feature_cols_numeric: List[str], feature_cols_cat: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols_numeric),
            ("cat", categorical_transformer, feature_cols_cat),
        ]
    )


def evaluate_model_cv(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
) -> Tuple[float, float]:
    """Évalue un modèle par validation croisée, retourne (mae, rmse)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_pred_cv = cross_val_predict(pipeline, X, y, cv=kf, n_jobs=-1)
    mae = mean_absolute_error(y, y_pred_cv)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred_cv)))
    print(f"[{name}] MAE CV = {mae:.3f} min, RMSE CV = {rmse:.3f} min")
    return mae, rmse


def infer_cluster_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Détecte automatiquement des features issues d'un clustering.

    - Labels de cluster (faible cardinalité) -> catégoriel (OneHotEncoder)
    - Colonnes de type proba/score de cluster -> numériques
    """
    reserved = {
        "duration_total_minutes",
        "airport",
        "alert_airport_id",
        "start_time",
        "end_time",
        "last_cloud_ground_time",
    }

    numeric_cluster_cols: List[str] = []
    cat_cluster_cols: List[str] = []

    # 1) Colonnes probas/scores (numériques)
    proba_tokens = ("proba", "prob", "score", "soft", "membership")
    for c in df.columns:
        cl = c.lower()
        if any(t in cl for t in proba_tokens) and ("cluster" in cl or "storm" in cl or "type" in cl):
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_cluster_cols.append(c)

    # 2) Colonne label unique (catégorielle)
    label_candidates: List[str] = []
    preferred_names = (
        "cluster",
        "clusters",
        "cluster_id",
        "cluster_label",
        "kmeans",
        "kmeans_label",
        "storm_type",
        "stormtype",
        "type_orage",
        "storm_cluster",
    )
    for c in df.columns:
        if c in reserved:
            continue
        cl = c.lower()
        if cl in preferred_names or any(cl.startswith(p + "_") for p in preferred_names) or "cluster" in cl:
            label_candidates.append(c)

    # Choisit le meilleur candidat label: faible cardinalité et non numérique continue
    best_label = None
    best_card = None
    for c in label_candidates:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and (s.dropna().astype(float) % 1 != 0).any():
            continue
        card = int(s.nunique(dropna=True))
        if card <= 1:
            continue
        if best_card is None or card < best_card:
            best_label = c
            best_card = card

    if best_label is not None:
        cat_cluster_cols.append(best_label)

    # Déduplique
    numeric_cluster_cols = sorted(set(numeric_cluster_cols))
    cat_cluster_cols = sorted(set(cat_cluster_cols))
    return numeric_cluster_cols, cat_cluster_cols


def main() -> None:
    base_dir = pathlib.Path(__file__).resolve().parent
    data_path_with_cluster = base_dir / "alerts_preprocessed_with_cluster.csv"
    data_path = base_dir / "alerts_preprocessed.csv"
    if data_path_with_cluster.exists():
        data_path = data_path_with_cluster
        print("Chargement avec clustering :", data_path.name)
    else:
        print("Fichier sans clustering. Pour en ajouter : python3 clustering_storm_types.py (ou script équivalent).")
    print(f"  Données : {data_path}")

    df = pd.read_csv(data_path)
    y = df["duration_total_minutes"].values

    base_numeric = [
        "n_lightnings",
        "n_cloud_ground",
        "n_intra_cloud",
        "mean_dist",
        "std_dist",
        "mean_amplitude",
        "start_year",
        "start_month",
        "start_dayofyear",
        "start_hour",
        "mean_maxis",
        "max_maxis",
        "density",
        "last_lightning_is_cg",
        "storm_size_km",
    ]
    feature_cols_numeric = [c for c in base_numeric if c in df.columns]
    feature_cols_cat = ["airport"] if "airport" in df.columns else []

    extra_numeric, extra_cat = infer_cluster_feature_columns(df)
    for c in extra_numeric:
        if c not in feature_cols_numeric:
            feature_cols_numeric.append(c)
    for c in extra_cat:
        if c not in feature_cols_cat:
            feature_cols_cat.append(c)

    if extra_numeric or extra_cat:
        print("  Features de clustering détectées :")
        if extra_cat:
            print(f"    - catégorielles: {extra_cat}")
        if extra_numeric:
            print(f"    - numériques: {extra_numeric}")

    # Cast des catégorielles en string pour OneHotEncoder
    for c in feature_cols_cat:
        df[c] = df[c].astype(str)

    X = df[feature_cols_numeric + feature_cols_cat].copy()
    preprocessor = build_preprocessor(feature_cols_numeric, feature_cols_cat)

    models: Dict[str, Pipeline] = {
        "rf_default": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=None,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "et_default": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=400,
                        max_depth=None,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gbr_default": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    if HAS_XGB:
        models["xgb_default"] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_cols_numeric, feature_cols_cat)),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=6,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    else:
        print("XGBoost non installé (pip install xgboost). Comparaison sans XGBoost.")

    results = []

    for name, pipe in models.items():
        print(f"\nÉvaluation du modèle de base : {name}")
        mae_cv, rmse_cv = evaluate_model_cv(name, pipe, X, y, n_splits=5)
        results.append({"model": name, "mae_cv": mae_cv, "rmse_cv": rmse_cv})

    # Tuning RF
    print("\nRecherche d'hyperparamètres pour RandomForest (rf_tuned) ...")
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    rf_param_dist = {
        "model__n_estimators": [200, 300, 400, 600],
        "model__max_depth": [None, 5, 8, 12, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.5],
    }
    search_rf = RandomizedSearchCV(
        rf_pipeline,
        param_distributions=rf_param_dist,
        n_iter=20,
        cv=5,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search_rf.fit(X, y)
    best_rf = search_rf.best_estimator_
    print(f"\nMeilleurs hyperparamètres RF : {search_rf.best_params_}")
    mae_cv_tuned, rmse_cv_tuned = evaluate_model_cv("rf_tuned", best_rf, X, y, n_splits=5)
    results.append({"model": "rf_tuned", "mae_cv": mae_cv_tuned, "rmse_cv": rmse_cv_tuned})

    # Tuning ET
    print("\nRecherche d'hyperparamètres pour Extra Trees (et_tuned) ...")
    et_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_cols_numeric, feature_cols_cat)),
            ("model", ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    et_param_dist = {
        "model__n_estimators": [300, 400, 600],
        "model__max_depth": [None, 12, 20],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.5],
    }
    search_et = RandomizedSearchCV(
        et_pipeline,
        param_distributions=et_param_dist,
        n_iter=15,
        cv=5,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search_et.fit(X, y)
    best_et = search_et.best_estimator_
    print(f"Meilleurs hyperparamètres ET : {search_et.best_params_}")
    mae_et_tuned, rmse_et_tuned = evaluate_model_cv("et_tuned", best_et, X, y, n_splits=5)
    results.append({"model": "et_tuned", "mae_cv": mae_et_tuned, "rmse_cv": rmse_et_tuned})

    # Tuning GBR
    print("\nRecherche d'hyperparamètres pour Gradient Boosting (gbr_tuned) ...")
    gbr_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_cols_numeric, feature_cols_cat)),
            ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
        ]
    )
    gbr_param_dist = {
        "model__n_estimators": [300, 500, 700],
        "model__max_depth": [3, 4, 5],
        "model__learning_rate": [0.03, 0.05, 0.08],
        "model__min_samples_leaf": [2, 4, 6],
    }
    search_gbr = RandomizedSearchCV(
        gbr_pipeline,
        param_distributions=gbr_param_dist,
        n_iter=15,
        cv=5,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search_gbr.fit(X, y)
    best_gbr = search_gbr.best_estimator_
    print(f"Meilleurs hyperparamètres GBR : {search_gbr.best_params_}")
    mae_gbr_tuned, rmse_gbr_tuned = evaluate_model_cv("gbr_tuned", best_gbr, X, y, n_splits=5)
    results.append({"model": "gbr_tuned", "mae_cv": mae_gbr_tuned, "rmse_cv": rmse_gbr_tuned})

    # Tuning XGB
    best_xgb = None
    if HAS_XGB:
        print("\nRecherche d'hyperparamètres pour XGBoost (xgb_tuned) ...")
        xgb_pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_cols_numeric, feature_cols_cat)),
                ("model", XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        )
        xgb_param_dist = {
            "model__n_estimators": [200, 400, 600, 800],
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__min_child_weight": [1, 3, 5],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
        }
        search_xgb = RandomizedSearchCV(
            xgb_pipeline,
            param_distributions=xgb_param_dist,
            n_iter=25,
            cv=5,
            scoring="neg_mean_absolute_error",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )
        search_xgb.fit(X, y)
        best_xgb = search_xgb.best_estimator_
        print(f"\nMeilleurs hyperparamètres XGBoost : {search_xgb.best_params_}")
        mae_xgb_tuned, rmse_xgb_tuned = evaluate_model_cv("xgb_tuned", best_xgb, X, y, n_splits=5)
        results.append({"model": "xgb_tuned", "mae_cv": mae_xgb_tuned, "rmse_cv": rmse_xgb_tuned})

    # Sélection du meilleur
    results_df = pd.DataFrame(results).sort_values("mae_cv")
    comparison_path = base_dir / "advanced_model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)
    print(f"\nTableau de comparaison des modèles sauvegardé dans : {comparison_path}")
    print(results_df)

    best_model_name = str(results_df.iloc[0]["model"])
    print(f"\nMeilleur modèle choisi : {best_model_name}")

    if best_model_name in models:
        best_model = models[best_model_name].fit(X, y)
    elif best_model_name == "rf_tuned":
        best_model = best_rf
    elif best_model_name == "et_tuned":
        best_model = best_et
    elif best_model_name == "gbr_tuned":
        best_model = best_gbr
    elif best_model_name == "xgb_tuned" and HAS_XGB and best_xgb is not None:
        best_model = best_xgb
    else:
        best_model = best_rf

    y_pred_all = best_model.predict(X)
    preds_df = X.copy()
    preds_df["duration_true"] = y
    preds_df["duration_pred_best"] = y_pred_all
    preds_path = base_dir / "advanced_model_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"Prédictions complètes du meilleur modèle sauvegardées dans : {preds_path}")


if __name__ == "__main__":
    main()

