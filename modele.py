import pathlib
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RandomizedSearchCV,
    cross_val_predict,
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

RANDOM_STATE = 42
BASE_DIR = pathlib.Path(__file__).resolve().parent
TARGET_CANDIDATES = ("duration_total_minutes", "duration_minutes")
# Evite le parallélisme imbriqué (RF n_jobs=-1 + CV n_jobs=-1) qui peut geler la machine.
CPU_COUNT = os.cpu_count() or 2
# Exécution volontairement séquentielle pour éviter les blocages joblib
# sur certaines machines (surcharge CPU/RAM quand CV et modèle parallélisent).
OUTER_N_JOBS = 1

GROUP_COL = "alert_airport_id"


def _oof_grouped_search_predict(
    search_template: RandomizedSearchCV,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> np.ndarray:
    """
    Prédictions out-of-fold : chaque pli val n'a jamais servi au tuning ni à l'entraînement.
    Outer split = GroupKFold (pas de fuite entre lignes d'une même alerte).
    Inner split du RandomizedSearchCV = GroupKFold sur le sous-jeu train du pli outer.
    """
    outer_cv = GroupKFold(n_splits=n_splits)
    y_oof = np.zeros(len(y), dtype=float)
    for train_idx, val_idx in outer_cv.split(X, y, groups):
        search = clone(search_template)
        search.fit(
            X.iloc[train_idx],
            y[train_idx],
            groups=groups[train_idx],
        )
        y_oof[val_idx] = search.predict(X.iloc[val_idx])
    return y_oof


def _resolve_csv_path(csv_path: str | None, use_enriched: bool) -> pathlib.Path:
    """Résout le CSV à utiliser pour la régression."""
    if csv_path:
        return pathlib.Path(csv_path)

    # Priorité au fichier généré par clustering.py
    clustered_path = BASE_DIR / "alerts_with_clusters.csv"
    if clustered_path.exists():
        return clustered_path

    if use_enriched:
        p = BASE_DIR / "alerts_final_model_17var.csv"
        if not p.exists():
            p = BASE_DIR / "alerts_final_model_enriched.csv"
        return p
    return BASE_DIR / "alerts_final_model.csv"


def _load_xy(
    csv_path: str | None,
    use_enriched: bool,
) -> tuple[pathlib.Path, pd.DataFrame, np.ndarray, np.ndarray | None, list[str]] | None:
    """Charge CSV, construit X (dummies), y, groupes. Retourne None si erreur."""
    csv_file = _resolve_csv_path(csv_path, use_enriched)
    if not csv_file.exists():
        print(f"Fichier absent : {csv_file}")
        if use_enriched:
            print("  Lancer d'abord : python3 enrich_csv.py")
        return None

    df = pd.read_csv(csv_file)
    if use_enriched and csv_path is None:
        print("  Utilisation du CSV enrichi – 17 variables (sous-ensemble optimal)\n")

    target_col = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target_col is None:
        print("Colonne cible manquante. Cibles acceptées : duration_total_minutes ou duration_minutes")
        return None

    y = df[target_col].values.astype(float)
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()

    if GROUP_COL in df.columns:
        groups = df[GROUP_COL].values
        n_grp = len(np.unique(groups))
        print(
            f"  Validation par groupe ({GROUP_COL}) : {n_grp} groupes, {len(y)} lignes "
            f"(GroupKFold 3 plis)."
        )
    else:
        groups = None
        print("  Colonne groupe absente : KFold classique (risque de fuite si lignes corrélées).")

    X = pd.get_dummies(X, drop_first=False)
    return csv_file, X, y, groups, feature_cols


def _build_mlp_pipeline() -> Pipeline:
    """
    MLP : sklearn attend X et y à échelle comparable.
    - StandardScaler sur les features
    - TransformedTargetRegressor + StandardScaler sur y (sinon descente de gradient très instable)
    """
    mlp = MLPRegressor(
        hidden_layer_sizes=(96, 48),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=1200,
        shuffle=True,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=30,
        random_state=RANDOM_STATE,
        tol=1e-4,
    )
    return Pipeline(
        [
            ("scaler_x", StandardScaler()),
            (
                "regressor",
                TransformedTargetRegressor(
                    regressor=mlp,
                    transformer=StandardScaler(),
                ),
            ),
        ]
    )


def run_mlp_only(csv_path: str | None = None, use_enriched: bool = False) -> None:
    """CV out-of-fold sur le MLP uniquement (rapide à lancer pour tester)."""
    print("Mode --mlp-only : MLP seul (OOF, même protocole que le pipeline complet).\n")
    loaded = _load_xy(csv_path, use_enriched)
    if loaded is None:
        return
    csv_file, X, y, groups, feature_cols = loaded
    cv_splits = 3
    kf_shuffle = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    pipe_mlp = _build_mlp_pipeline()
    print("  MLP (X + y standardisés, TransformedTargetRegressor)…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        if groups is not None:
            y_pred = cross_val_predict(
                pipe_mlp,
                X,
                y,
                cv=GroupKFold(n_splits=cv_splits),
                groups=groups,
                n_jobs=OUTER_N_JOBS,
            )
        else:
            y_pred = cross_val_predict(
                pipe_mlp, X, y, cv=kf_shuffle, n_jobs=OUTER_N_JOBS
            )
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"\n  Fichier : {csv_file.name}")
    print(f"  Lignes : {len(y)}  |  Colonnes features (dummies) : {X.shape[1]}")
    print(f"  mlp_dense  MAE = {mae:.3f} min   RMSE = {rmse:.3f} min")


def run_model(csv_path: str | None = None, use_enriched: bool = False) -> None:
    loaded = _load_xy(csv_path, use_enriched)
    if loaded is None:
        return
    csv_file, X, y, groups, feature_cols = loaded

    cv_splits = 3
    kf_shuffle = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    results = []

    # Baseline : régression linéaire (OLS) sur les mêmes features, même protocole OOF / groupes
    print("  Régression linéaire (baseline)")
    pipe_linear = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    if groups is not None:
        gkf_lin = GroupKFold(n_splits=cv_splits)
        y_pred_lr = cross_val_predict(
            pipe_linear, X, y, cv=gkf_lin, groups=groups, n_jobs=OUTER_N_JOBS
        )
    else:
        y_pred_lr = cross_val_predict(
            pipe_linear, X, y, cv=kf_shuffle, n_jobs=OUTER_N_JOBS
        )
    lr_mae = mean_absolute_error(y, y_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y, y_pred_lr))
    results.append(("linear_baseline", lr_mae, lr_rmse))
    print(f"    -> linear_baseline terminé | MAE={lr_mae:.3f} RMSE={lr_rmse:.3f}")

    # RF défaut MODELE 1

    print("  Random forest default MODELE 1")
    rf_estimators = 120
    pipe_rf = Pipeline([
        ("model", RandomForestRegressor(n_estimators=rf_estimators, max_depth=None, random_state=RANDOM_STATE, n_jobs=1)),
    ])
    if groups is not None:
        gkf = GroupKFold(n_splits=cv_splits)
        y_pred = cross_val_predict(
            pipe_rf, X, y, cv=gkf, groups=groups, n_jobs=OUTER_N_JOBS
        )
    else:
        y_pred = cross_val_predict(pipe_rf, X, y, cv=kf_shuffle, n_jobs=OUTER_N_JOBS)
    rf_default_mae = mean_absolute_error(y, y_pred)
    rf_default_rmse = np.sqrt(mean_squared_error(y, y_pred))
    results.append(("rf_default", rf_default_mae, rf_default_rmse))  # calcul du MAE et RMSE
    print(f"    -> rf_default terminé | MAE={rf_default_mae:.3f} RMSE={rf_default_rmse:.3f}")

    # Random forest tuned MODELE 2

    print("  Random forest tuned MODELE 2")
    pipe_tuned = Pipeline([
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)),
    ])
    inner_cv_rf: GroupKFold | KFold
    if groups is not None:
        inner_cv_rf = GroupKFold(n_splits=cv_splits)
    else:
        inner_cv_rf = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    search_rf_template = RandomizedSearchCV(
        pipe_tuned,
        param_distributions={
            "model__n_estimators": [120, 200, 300],
            "model__max_depth": [None, 15, 25],
            "model__min_samples_leaf": [1, 2, 3],
            "model__max_features": ["sqrt", 0.4, 0.6],
        },
        n_iter=6,
        cv=inner_cv_rf,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=OUTER_N_JOBS,
        verbose=0,
    )
    if groups is not None:
        y_pred_tuned = _oof_grouped_search_predict(
            search_rf_template, X, y, groups, cv_splits
        )
    else:
        # Nested CV : RandomizedSearchCV ré-entraîné sur chaque pli train uniquement
        y_pred_tuned = cross_val_predict(
            search_rf_template, X, y, cv=kf_shuffle, n_jobs=OUTER_N_JOBS
        )
    rf_tuned_mae = mean_absolute_error(y, y_pred_tuned)
    rf_tuned_rmse = np.sqrt(mean_squared_error(y, y_pred_tuned))
    results.append(("rf_tuned", rf_tuned_mae, rf_tuned_rmse))
    print(f"    -> rf_tuned terminé | MAE={rf_tuned_mae:.3f} RMSE={rf_tuned_rmse:.3f}")

    # XGBoost tuné si dispo MODELE 3

    print("  XGBoost tuned MODELE 3")

    y_pred_xgb: np.ndarray | None = None
    if HAS_XGB:
        print("  Tuning XGBoost (OOF + groupes si disponibles)...")
        pipe_xgb = Pipeline([
            ("model", XGBRegressor(random_state=RANDOM_STATE, n_jobs=1)),
        ])
        inner_cv_xgb: GroupKFold | KFold = (
            GroupKFold(n_splits=cv_splits)
            if groups is not None
            else KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
        )
        search_xgb_template = RandomizedSearchCV(
            pipe_xgb,
            param_distributions={
                "model__n_estimators": [120, 200, 300],
                "model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__min_child_weight": [1, 3, 5],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
            n_iter=6,
            cv=inner_cv_xgb,
            scoring="neg_mean_absolute_error",
            random_state=RANDOM_STATE,
            n_jobs=OUTER_N_JOBS,
            verbose=0,
        )
        if groups is not None:
            y_pred_xgb = _oof_grouped_search_predict(
                search_xgb_template, X, y, groups, cv_splits
            )
        else:
            y_pred_xgb = cross_val_predict(
                search_xgb_template, X, y, cv=kf_shuffle, n_jobs=OUTER_N_JOBS
            )
        xgb_tuned_mae = mean_absolute_error(y, y_pred_xgb)
        xgb_tuned_rmse = np.sqrt(mean_squared_error(y, y_pred_xgb))
        results.append(("xgb_tuned", xgb_tuned_mae, xgb_tuned_rmse))
        print(f"    -> xgb_tuned terminé | MAE={xgb_tuned_mae:.3f} RMSE={xgb_tuned_rmse:.3f}")
    else:
        print("  (XGBoost non installé : pip install xgboost)")

    # CatBoost tuné si dispo (même X numérique après get_dummies que les autres modèles)
    print("  CatBoost tuned MODELE 4")

    y_pred_cat: np.ndarray | None = None
    if HAS_CATBOOST:
        print("  Tuning CatBoost (OOF + groupes si disponibles)...")
        pipe_cat = Pipeline(
            [
                (
                    "model",
                    CatBoostRegressor(
                        random_seed=RANDOM_STATE,
                        verbose=False,
                        loss_function="MAE",
                        thread_count=1,
                    ),
                ),
            ]
        )
        inner_cv_cat: GroupKFold | KFold = (
            GroupKFold(n_splits=cv_splits)
            if groups is not None
            else KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
        )
        search_cat_template = RandomizedSearchCV(
            pipe_cat,
            param_distributions={
                "model__iterations": [400, 600, 900],
                "model__depth": [4, 6, 8, 10],
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__l2_leaf_reg": [1, 3, 7, 10],
                "model__subsample": [0.8, 1.0],
                "model__border_count": [128, 254],
            },
            n_iter=6,
            cv=inner_cv_cat,
            scoring="neg_mean_absolute_error",
            random_state=RANDOM_STATE,
            n_jobs=OUTER_N_JOBS,
            verbose=0,
        )
        if groups is not None:
            y_pred_cat = _oof_grouped_search_predict(
                search_cat_template, X, y, groups, cv_splits
            )
        else:
            y_pred_cat = cross_val_predict(
                search_cat_template, X, y, cv=kf_shuffle, n_jobs=OUTER_N_JOBS
            )
        cat_mae = mean_absolute_error(y, y_pred_cat)
        cat_rmse = np.sqrt(mean_squared_error(y, y_pred_cat))
        results.append(("catboost_tuned", cat_mae, cat_rmse))
        print(f"    -> catboost_tuned terminé | MAE={cat_mae:.3f} RMSE={cat_rmse:.3f}")
    else:
        print("  (CatBoost non installé : pip install catboost)")

    # MLP : X et y standardisés (voir _build_mlp_pipeline)
    print("  MLP (réseau dense sklearn, X+y scalés)")
    pipe_mlp = _build_mlp_pipeline()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        if groups is not None:
            gkf_mlp = GroupKFold(n_splits=cv_splits)
            y_pred_mlp = cross_val_predict(
                pipe_mlp, X, y, cv=gkf_mlp, groups=groups, n_jobs=OUTER_N_JOBS
            )
        else:
            y_pred_mlp = cross_val_predict(
                pipe_mlp, X, y, cv=kf_shuffle, n_jobs=OUTER_N_JOBS
            )
    mlp_mae = mean_absolute_error(y, y_pred_mlp)
    mlp_rmse = np.sqrt(mean_squared_error(y, y_pred_mlp))
    results.append(("mlp_dense", mlp_mae, mlp_rmse))
    print(f"    -> mlp_dense terminé | MAE={mlp_mae:.3f} RMSE={mlp_rmse:.3f}")

    print("\n" + "=" * 55)
    print("  " + csv_file.name + f" – Résultats (CV {cv_splits}-fold)")
    print("=" * 55)
    print(f"  Alertes : {len(y)}  |  Variables : {len(feature_cols)}")
    print()

    # Choix du meilleur modèle (critère : MAE la plus faible en CV)
    best_name, best_mae, best_rmse = min(results, key=lambda x: x[1])

    labels_fr = {
        "linear_baseline": "Régression linéaire (baseline OLS + scaling)",
        "rf_default": "Random Forest (paramètres par défaut)",
        "rf_tuned": "Random Forest (hyperparamètres optimisés)",
        "xgb_tuned": "XGBoost (hyperparamètres optimisés)",
        "catboost_tuned": "CatBoost (hyperparamètres optimisés)",
        "mlp_dense": "MLP (dense, X+y standardisés)",
    }

    for name, mae, rmse in sorted(results, key=lambda x: x[1]):
        sous_10 = "  ✓ sous 10 min !" if mae < 10 else ""
        print(f"  {name:18s}  MAE = {mae:.3f} min   RMSE = {rmse:.3f} min{sous_10}")
    print("=" * 55)
    print(
        f"  Modèle retenu : {best_name} — {labels_fr.get(best_name, best_name)} "
        f"(MAE = {best_mae:.3f} min, RMSE = {best_rmse:.3f} min)"
    )
    print("=" * 55)
    if best_mae < 10:
        print("  Objectif < 10 min MAE atteint.")
    else:
        print(f"  Meilleur MAE : {best_mae:.2f} min. Descendre sous 10 min peut être limité par la variabilité météo.")

    # Prédictions pour probabilite_par_minute.py : uniquement OOF (jamais fit sur tout X puis predict(X))
    if best_name == "linear_baseline":
        y_pred_final = y_pred_lr
    elif best_name == "rf_default":
        y_pred_final = y_pred
    elif best_name == "rf_tuned":
        y_pred_final = y_pred_tuned
    elif best_name == "xgb_tuned" and y_pred_xgb is not None:
        y_pred_final = y_pred_xgb
    elif best_name == "catboost_tuned" and y_pred_cat is not None:
        y_pred_final = y_pred_cat
    elif best_name == "mlp_dense":
        y_pred_final = y_pred_mlp
    else:
        y_pred_final = y_pred

    preds_path = BASE_DIR / "advanced_model_predictions.csv"
    pd.DataFrame(
        {
            "duration_true": y.astype(float),
            "duration_pred_best": y_pred_final.astype(float),
        }
    ).to_csv(preds_path, index=False)
    print(
        f"\n  Prédictions sauvegardées (out-of-fold, sans fuite train→test) : {preds_path.name}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Régression durée d'alerte.")
    parser.add_argument("--csv", type=str, default=None, help="Chemin CSV à utiliser.")
    parser.add_argument(
        "--enriched",
        action="store_true",
        help="Utilise le CSV enrichi (si --csv non fourni).",
    )
    parser.add_argument(
        "--mlp-only",
        action="store_true",
        help="N'évalue que le MLP (CV OOF), pour test rapide.",
    )
    args = parser.parse_args()
    if args.mlp_only:
        run_mlp_only(csv_path=args.csv, use_enriched=args.enriched)
    else:
        run_model(csv_path=args.csv, use_enriched=args.enriched)