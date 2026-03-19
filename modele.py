import pathlib
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

    # Rend le script compatible avec des colonnes catégorielles (ex: airport, storm_type)
    X = pd.get_dummies(X, drop_first=False)

    cv_splits = 3
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    results = []

    # RF défaut MODELE 1

    print("  Random forest default MODELE 1")
    rf_estimators = 120
    pipe_rf = Pipeline([
        ("model", RandomForestRegressor(n_estimators=rf_estimators, max_depth=None, random_state=RANDOM_STATE, n_jobs=1)),
    ])
    y_pred = cross_val_predict(pipe_rf, X, y, cv=kf, n_jobs=OUTER_N_JOBS)
    rf_default_mae = mean_absolute_error(y, y_pred)
    rf_default_rmse = np.sqrt(mean_squared_error(y, y_pred))
    results.append(("rf_default", rf_default_mae, rf_default_rmse))  # calcul du MAE et RMSE
    print(f"    -> rf_default terminé | MAE={rf_default_mae:.3f} RMSE={rf_default_rmse:.3f}")

    # Random forest tuned MODELE 2


    print("  Random forest tuned MODELE 2")
    pipe_tuned = Pipeline([
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)),
    ])
    search = RandomizedSearchCV(
        pipe_tuned,
        param_distributions={
            "model__n_estimators": [120, 200, 300],
            "model__max_depth": [None, 15, 25],
            "model__min_samples_leaf": [1, 2, 3],
            "model__max_features": ["sqrt", 0.4, 0.6],
        },
        n_iter=6,
        cv=cv_splits,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=OUTER_N_JOBS,
        verbose=0,
    )
    search.fit(X, y)
    y_pred_tuned = cross_val_predict(search.best_estimator_, X, y, cv=kf, n_jobs=OUTER_N_JOBS)
    rf_tuned_mae = mean_absolute_error(y, y_pred_tuned)
    rf_tuned_rmse = np.sqrt(mean_squared_error(y, y_pred_tuned))
    results.append(("rf_tuned", rf_tuned_mae, rf_tuned_rmse))
    print(f"    -> rf_tuned terminé | MAE={rf_tuned_mae:.3f} RMSE={rf_tuned_rmse:.3f}")

    # XGBoost tuné si dispo MODELE 3

    print("  XGBoost tuned MODELE 3")

    if HAS_XGB:
        print("  Tuning XGBoost...")
        pipe_xgb = Pipeline([
            ("model", XGBRegressor(random_state=RANDOM_STATE, n_jobs=1)),
        ])
        search_xgb = RandomizedSearchCV(
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
            cv=cv_splits,
            scoring="neg_mean_absolute_error",
            random_state=RANDOM_STATE,
            n_jobs=OUTER_N_JOBS,
            verbose=0,
        )
        search_xgb.fit(X, y)
        y_pred_xgb = cross_val_predict(search_xgb.best_estimator_, X, y, cv=kf, n_jobs=OUTER_N_JOBS)
        results.append(("xgb_tuned", mean_absolute_error(y, y_pred_xgb), np.sqrt(mean_squared_error(y, y_pred_xgb))))
    else:
        print("  (XGBoost non installé : pip install xgboost)")
    print("\n" + "=" * 55)
    print("  " + csv_file.name + f" – Résultats (CV {cv_splits}-fold)")
    print("=" * 55)
    print(f"  Alertes : {len(y)}  |  Variables : {len(feature_cols)}")
    print()

    # Choix du meilleur modele

    best_mae = min(r[1] for r in results)

    for name, mae, rmse in sorted(results, key=lambda x: x[1]):
        sous_10 = "  ✓ sous 10 min !" if mae < 10 else ""
        print(f"  {name:12s}  MAE = {mae:.3f} min   RMSE = {rmse:.3f} min{sous_10}")
    print("=" * 55)
    if best_mae < 10:
        print("  Objectif < 10 min MAE atteint.")#si modele avec MAE < 10 min
    else:
        print(f"  Meilleur MAE : {best_mae:.2f} min. Descendre sous 10 min peut être limité par la variabilité météo.")


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