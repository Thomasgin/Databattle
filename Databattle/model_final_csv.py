"""
Data Battle 2026 – Modèle adapté au nouveau CSV (alerts_final_model.csv).
Variables décrites dans INFO_VAR.pdf. Cible : duration_total_minutes.
Affiche MAE et RMSE en validation croisée. Teste RF, RF tuné, XGBoost pour viser < 10 min MAE.
"""
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocvessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RANDOM_STATE = 42
BASE_DIR = pathlib.Path(__file__).resolve().parent


def main():

    # Chargement du fichier csv avec les vars enrichies ou pas
    import sys
    use_enriched = "enriched" in sys.argv
    if use_enriched:
        # Par défaut : 17 variables (sous-ensemble RFECV, meilleur compromis)
        CSV_PATH = BASE_DIR / "alerts_final_model_17var.csv"
        if not CSV_PATH.exists():
            CSV_PATH = BASE_DIR / "alerts_final_model_enriched.csv"
    else:
        CSV_PATH = BASE_DIR / "alerts_final_model.csv"
    if not CSV_PATH.exists():v
        print(f"Fichier absent : {CSV_PATH}")
        if use_enriched:
            print("  Lancer d'abord : python3 enrich_csv.py")
        return

    df = pd.read_csv(CSV_PATH)
    if use_enriched:
        print("  Utilisation du CSV enrichi – 17 variables (sous-ensemble optimal)\n")

    # Séparation variables cible et explicative pour entrainé le modele sur X et Y
    y = df["duration_total_minutes"].values
    feature_cols = [c for c in df.columns if c != "duration_total_minutes"]
    X = df[feature_cols]
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []

    ## Random forest par default MODELE 1


    pipe_rf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    # ypred sont les preiction de la duree total de l'orage pour chaque alerte
    y_pred = cross_val_predict(pipe_rf, X, y, cv=kf, n_jobs=-1)
    #Ajout des resultats
    results.append(("rf_default", mean_absolute_error(y, y_pred),# MAE
     np.sqrt(mean_squared_error(y, y_pred))))#RMSE

    ## Random Forest tuned MODELE 2


    print("  Tuning RF...")
    pipe_tuned = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    search = RandomizedSearchCV( # Recherche des hyper parametres
        pipe_tuned,
        param_distributions={
            "model__n_estimators": [400, 600, 800],
            "model__max_depth": [None, 15, 25],
            "model__min_samples_leaf": [1, 2, 3],
            "model__max_features": ["sqrt", 0.4, 0.6],
        },
        n_iter=25,
        cv=5,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)
    y_pred_tuned = cross_val_predict(search.best_estimator_, X, y, cv=kf, n_jobs=-1) #meilleur modele trouvé avec search.best_estimator_
    #Ajout des resultats
    results.append(("rf_tuned", mean_absolute_error(y, y_pred_tuned), np.sqrt(mean_squared_error(y, y_pred_tuned))))

    # XGBoost tuné si dispo MODELE 3
    if HAS_XGB:
        print("  Tuning XGBoost...")
        pipe_xgb = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        ])
        search_xgb = RandomizedSearchCV(
            pipe_xgb,
            param_distributions={
                "model__n_estimators": [400, 600, 800],
                "model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__min_child_weight": [1, 3, 5],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
            n_iter=20,
            cv=5,
            scoring="neg_mean_absolute_error",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        search_xgb.fit(X, y)
        y_pred_xgb = cross_val_predict(search_xgb.best_estimator_, X, y, cv=kf, n_jobs=-1)
        results.append(("xgb_tuned", mean_absolute_error(y, y_pred_xgb), np.sqrt(mean_squared_error(y, y_pred_xgb))))
    else:
        print("  (XGBoost non installé : pip install xgboost)")

    # Affichage
    print("\n" + "=" * 55)
    print("  " + CSV_PATH.name + " – Résultats (CV 5-fold)")
    print("=" * 55)
    print(f"  Alertes : {len(y)}  |  Variables : {len(feature_cols)}")CSV_PATH
        print(f"  Meilleur MAE : {best_mae:.2f} min. Descendre sous 10 min peut être limité par la variabilité météo.")


if __name__ == "__main__":
    main()
