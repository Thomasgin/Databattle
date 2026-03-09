import pathlib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main() -> None:
    """Modèle de base pour prédire la durée totale d'une alerte."""
    base_dir = pathlib.Path(__file__).resolve().parent
    data_path = base_dir / "alerts_preprocessed.csv"

    print(f"Chargement des données agrégées depuis : {data_path}")
    df = pd.read_csv(data_path, parse_dates=["start_time", "end_time", "last_cloud_ground_time"])

    # Cible : durée totale en minutes
    y = df["duration_total_minutes"].values

    # Features de base
    feature_cols_numeric = [
        "n_lightnings",
        "n_cloud_ground",
        "n_intra_cloud",
        "start_year",
        "start_month",
        "start_dayofyear",
        "start_hour",
    ]
    feature_cols_cat = ["airport"]

    X = df[feature_cols_numeric + feature_cols_cat].copy()

    # séparation entraînement / validation (aléatoire simple pour ce premier modèle)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("Entraînement du modèle RandomForestRegressor...")
    clf.fit(X_train, y_train)

    print("Évaluation sur le jeu de validation...")
    y_pred = clf.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    print(f"MAE (minutes)  : {mae:.3f}")
    print(f"RMSE (minutes) : {rmse:.3f}")

    # Estimation simple de la variance résiduelle pour une interprétation probabiliste
    residuals = y_val - y_pred
    sigma = residuals.std()
    print(f"Écart-type des résidus (sigma) ≈ {sigma:.3f} minutes")

    # Sauvegarde des prédictions de validation pour analyse ultérieure
    preds_df = X_val.copy()
    preds_df["duration_true"] = y_val
    preds_df["duration_pred"] = y_pred
    preds_df["residual"] = residuals

    preds_path = base_dir / "model_validation_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"Prédictions de validation sauvegardées dans : {preds_path}")


if __name__ == "__main__":
    main()

