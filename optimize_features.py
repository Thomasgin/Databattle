"""
Teste des combinaisons de variables pour minimiser MAE/RMSE.
1. Charge le CSV enrichi (lancer enrich_csv.py avant).
2. RFECV : sélection récursive des meilleures variables (validation croisée).
3. Teste aussi : toutes les variables, puis meilleur sous-ensemble trouvé.
4. Compare RF et XGBoost sur le sous-ensemble optimal.
"""
import pathlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
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
CSV_PATH = BASE_DIR / "alerts_final_model_enriched.csv"


def train_evaluate(X, y, name, kf):
    """RF tuné + XGB tuné sur X,y, retourne (best_name, best_mae, best_rmse)."""
    results = []
    # RF
    pipe_rf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    search_rf = RandomizedSearchCV(
        pipe_rf,
        param_distributions={
            "model__n_estimators": [400, 600],
            "model__max_depth": [None, 20],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", 0.5],
        },
        n_iter=12,
        cv=5,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search_rf.fit(X, y)
    pred_rf = cross_val_predict(search_rf.best_estimator_, X, y, cv=kf, n_jobs=-1)
    results.append((f"{name}_rf", mean_absolute_error(y, pred_rf), np.sqrt(mean_squared_error(y, pred_rf))))
    # XGB
    if HAS_XGB:
        pipe_xgb = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        ])
        search_xgb = RandomizedSearchCV(
            pipe_xgb,
            param_distributions={
                "model__n_estimators": [400, 600],
                "model__max_depth": [4, 6],
                "model__learning_rate": [0.03, 0.05],
                "model__subsample": [0.8, 1.0],
            },
            n_iter=10,
            cv=5,
            scoring="neg_mean_absolute_error",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        search_xgb.fit(X, y)
        pred_xgb = cross_val_predict(search_xgb.best_estimator_, X, y, cv=kf, n_jobs=-1)
        results.append((f"{name}_xgb", mean_absolute_error(y, pred_xgb), np.sqrt(mean_squared_error(y, pred_xgb))))
    return results


def main():
    if not CSV_PATH.exists():
        print(f"Fichier absent : {CSV_PATH}. Lancer : python3 enrich_csv.py")
        return

    df = pd.read_csv(CSV_PATH)
    y = df["duration_total_minutes"].values
    feature_cols = [c for c in df.columns if c != "duration_total_minutes"]
    X = df[feature_cols]
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("=" * 60)
    print("  Optimisation variables – RFECV + comparaison combinaisons")
    print("=" * 60)
    print(f"  Alertes : {len(y)}  |  Variables initiales : {len(feature_cols)}")
    print()

    # 1) Résultat avec TOUTES les variables
    print("  1) Modèle avec toutes les variables...")
    all_results = train_evaluate(X, y, "all", kf)
    for name, mae, rmse in all_results:
        print(f"     {name}  MAE = {mae:.3f}  RMSE = {rmse:.3f}")
    best_so_far = min(r[1] for r in all_results)
    best_name_so_far = [r[0] for r in all_results if r[1] == best_so_far][0]

    # 2) RFECV : meilleur sous-ensemble de variables (scale avant pour stabilité)
    print("\n  2) RFECV : sélection des meilleures variables (CV 5-fold)...")
    X_scaled = StandardScaler().fit_transform(X)
    selector = RFECV(
        estimator=RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
        step=1,
        cv=kf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        min_features_to_select=10,
    )
    selector.fit(X_scaled, y)
    selected_mask = selector.support_
    n_selected = int(selected_mask.sum())
    selected_cols = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]
    X_sel = X[selected_cols]

    print(f"     Nombre de variables sélectionnées : {n_selected} / {len(feature_cols)}")
    print(f"     Variables gardées : {selected_cols}")

    # 3) Réentraîner RF + XGB sur le sous-ensemble sélectionné
    print("\n  3) Modèle sur le sous-ensemble RFECV...")
    sel_results = train_evaluate(X_sel, y, "selected", kf)
    for name, mae, rmse in sel_results:
        print(f"     {name}  MAE = {mae:.3f}  RMSE = {rmse:.3f}")
        if mae < best_so_far:
            best_so_far = mae
            best_name_so_far = name

    # 4) Résumé
    print("\n" + "=" * 60)
    print("  MEILLEUR RÉSULTAT")
    print("=" * 60)
    print(f"  Meilleur : {best_name_so_far}  |  MAE min = {best_so_far:.3f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
