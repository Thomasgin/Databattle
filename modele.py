import pathlib
import os

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RandomizedSearchCV,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

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


def run_model(csv_path: str | None = None, use_enriched: bool = False) -> None:
    csv_file = _resolve_csv_path(csv_path, use_enriched)
    if not csv_file.exists():
        print(f"Fichier absent : {csv_file}")
        if use_enriched:
            print("  Lancer d'abord : python3 enrich_csv.py")
        return

    df = pd.read_csv(csv_file)
    if use_enriched and csv_path is None:
        print("  Utilisation du CSV enrichi – 17 variables (sous-ensemble optimal)\n")

    target_col = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target_col is None:
        print("Colonne cible manquante. Cibles acceptées : duration_total_minutes ou duration_minutes")
        return

    y = df[target_col].values
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()

    # Groupes pour éviter que des lignes d'une même alerte soient éclatées train/val
    groups: np.ndarray | None
    if GROUP_COL in df.columns:
        groups = df[GROUP_COL].values
        n_grp = len(np.unique(groups))
        print(
            f"  Validation par groupe ({GROUP_COL}) : {n_grp} groupes, {len(y)} lignes "
            f"(GroupKFold {3} plis)."
        )
    else:
        groups = None
        print("  Colonne groupe absente : KFold classique (risque de fuite si lignes corrélées).")

    # Rend le script compatible avec des colonnes catégorielles (ex: airport, storm_type)
    X = pd.get_dummies(X, drop_first=False)

    cv_splits = 3
    kf_shuffle = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    results = []

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
    print("\n" + "=" * 55)
    print("  " + csv_file.name + f" – Résultats (CV {cv_splits}-fold)")
    print("=" * 55)
    print(f"  Alertes : {len(y)}  |  Variables : {len(feature_cols)}")
    print()

    # Choix du meilleur modèle (critère : MAE la plus faible en CV)
    best_name, best_mae, best_rmse = min(results, key=lambda x: x[1])

    labels_fr = {
        "rf_default": "Random Forest (paramètres par défaut)",
        "rf_tuned": "Random Forest (hyperparamètres optimisés)",
        "xgb_tuned": "XGBoost (hyperparamètres optimisés)",
    }

    for name, mae, rmse in sorted(results, key=lambda x: x[1]):
        sous_10 = "  ✓ sous 10 min !" if mae < 10 else ""
        print(f"  {name:12s}  MAE = {mae:.3f} min   RMSE = {rmse:.3f} min{sous_10}")
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
    if best_name == "rf_default":
        y_pred_final = y_pred
    elif best_name == "rf_tuned":
        y_pred_final = y_pred_tuned
    elif best_name == "xgb_tuned" and y_pred_xgb is not None:
        y_pred_final = y_pred_xgb
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
    args = parser.parse_args()
    run_model(csv_path=args.csv, use_enriched=args.enriched)