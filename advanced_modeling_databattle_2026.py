import pathlib
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


def build_preprocessor(feature_cols_numeric, feature_cols_cat):
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols_numeric),
            ("cat", categorical_transformer, feature_cols_cat),
        ]
    )
    return preprocessor


def evaluate_model_cv(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
) -> Tuple[float, float]:
    """Évalue un modèle par validation croisée, retourne (mae_moyenne, rmse_moyenne)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # cross_val_predict pour avoir des prédictions "out-of-fold"
    y_pred_cv = cross_val_predict(pipeline, X, y, cv=kf, n_jobs=-1)

    mae = mean_absolute_error(y, y_pred_cv)
    mse = mean_squared_error(y, y_pred_cv)
    rmse = np.sqrt(mse)
    print(f"[{name}] MAE CV = {mae:.3f} min, RMSE CV = {rmse:.3f} min")
    return mae, rmse


def main() -> None:
    base_dir = pathlib.Path(__file__).resolve().parent
    data_path = base_dir / "alerts_preprocessed.csv"

    print(f"Chargement des données agrégées depuis : {data_path}")
    df = pd.read_csv(data_path)

    y = df["duration_total_minutes"].values

    feature_cols_numeric = [
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
    ]
    feature_cols_cat = ["airport"]

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
        # Préprocesseur dédié pour XGBoost (même config)
        preprocessor_xgb = build_preprocessor(feature_cols_numeric, feature_cols_cat)
        models["xgb_default"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor_xgb),
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

    # Évaluation des modèles de base
    for name, pipe in models.items():
        print(f"\nÉvaluation du modèle de base : {name}")
        mae_cv, rmse_cv = evaluate_model_cv(name, pipe, X, y, n_splits=5)
        results.append({"model": name, "mae_cv": mae_cv, "rmse_cv": rmse_cv})

    # RandomizedSearchCV sur RandomForest pour affiner
    print("\nRecherche d'hyperparamètres pour RandomForest (rf_tuned) ...")
    rf_base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    rf_param_dist = {
        "model__n_estimators": [200, 300, 400, 600],
        "model__max_depth": [None, 5, 8, 12, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.5],
    }

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", rf_base),
        ]
    )

    search = RandomizedSearchCV(
        rf_pipeline,
        param_distributions=rf_param_dist,
        n_iter=20,
        cv=5,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X, y)
    best_rf = search.best_estimator_
    print(f"\nMeilleurs hyperparamètres RF : {search.best_params_}")

    mae_cv_tuned, rmse_cv_tuned = evaluate_model_cv("rf_tuned", best_rf, X, y, n_splits=5)
    results.append({"model": "rf_tuned", "mae_cv": mae_cv_tuned, "rmse_cv": rmse_cv_tuned})

    # Extra Trees réglé (bon compromis précision / complexité)
    print("\nRecherche d'hyperparamètres pour Extra Trees (et_tuned) ...")
    et_base = ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    et_param_dist = {
        "model__n_estimators": [300, 400, 600],
        "model__max_depth": [None, 12, 20],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.5],
    }
    et_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(feature_cols_numeric, feature_cols_cat)),
        ("model", et_base),
    ])
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

    # Gradient Boosting réglé (alternatif compact)
    print("\nRecherche d'hyperparamètres pour Gradient Boosting (gbr_tuned) ...")
    gbr_base = GradientBoostingRegressor(random_state=RANDOM_STATE)
    gbr_param_dist = {
        "model__n_estimators": [300, 500, 700],
        "model__max_depth": [3, 4, 5],
        "model__learning_rate": [0.03, 0.05, 0.08],
        "model__min_samples_leaf": [2, 4, 6],
    }
    gbr_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(feature_cols_numeric, feature_cols_cat)),
        ("model", gbr_base),
    ])
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

    # RandomizedSearchCV sur XGBoost si disponible
    if HAS_XGB:
        print("\nRecherche d'hyperparamètres pour XGBoost (xgb_tuned) ...")
        preprocessor_xgb2 = build_preprocessor(feature_cols_numeric, feature_cols_cat)
        xgb_base = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        xgb_param_dist = {
            "model__n_estimators": [200, 400, 600, 800],
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__min_child_weight": [1, 3, 5],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
        }
        xgb_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor_xgb2),
                ("model", xgb_base),
            ]
        )
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

    # Comparaison et sélection du meilleur modèle (plus petite MAE)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mae_cv")
    comparison_path = base_dir / "advanced_model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)
    print(f"\nTableau de comparaison des modèles sauvegardé dans : {comparison_path}")
    print(results_df)

    best_model_name = results_df.iloc[0]["model"]
    print(f"\nMeilleur modèle choisi : {best_model_name} (Data Battle – meilleur MAE, utilisé pour probability_per_minute et comparaison avant/après)")

    if best_model_name == "rf_default":
        best_model = models["rf_default"].fit(X, y)
    elif best_model_name == "et_default":
        best_model = models["et_default"].fit(X, y)
    elif best_model_name == "gbr_default":
        best_model = models["gbr_default"].fit(X, y)
    elif best_model_name == "xgb_default" and HAS_XGB:
        best_model = models["xgb_default"].fit(X, y)
    elif best_model_name == "xgb_tuned" and HAS_XGB:
        best_model = best_xgb
    elif best_model_name == "et_tuned":
        best_model = best_et
    elif best_model_name == "gbr_tuned":
        best_model = best_gbr
    else:
        best_model = best_rf  # rf_tuned

    # Prédictions finales du meilleur modèle sur tout X
    y_pred_all = best_model.predict(X)
    preds_df = X.copy()
    preds_df["duration_true"] = y
    preds_df["duration_pred_best"] = y_pred_all

    preds_path = base_dir / "advanced_model_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"Prédictions complètes du meilleur modèle sauvegardées dans : {preds_path}")


if __name__ == "__main__":
    main()

