import pathlib

import numpy as np
import pandas as pd
from math import erf, sqrt


def normal_cdf(x: np.ndarray) -> np.ndarray:
    """CDF de la loi normale standard, implémentée via erf."""
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def main() -> None:
    """Calcule des probabilités de fin d'alerte pour différents horizons."""
    base_dir = pathlib.Path(__file__).resolve().parent
    preds_path = base_dir / "model_validation_predictions.csv"

    print(f"Chargement des prédictions de validation depuis : {preds_path}")
    df = pd.read_csv(preds_path)

    # On récupère la prédiction moyenne et les résidus
    y_pred = df["duration_pred"].values
    residuals = df["residual"].values

    sigma = residuals.std()
    print(f"Écart-type des résidus recalculé (sigma) ≈ {sigma:.3f} minutes")

    horizons = [10.0, 20.0, 30.0, 45.0, 60.0]  # en minutes

    for t in horizons:
        z = (t - y_pred) / sigma
        p = normal_cdf(z)
        col_name = f"p_end_before_{int(t)}"
        df[col_name] = p
        print(f"Ajout de la colonne {col_name}")

    out_path = base_dir / "validation_probabilities.csv"
    df.to_csv(out_path, index=False)
    print(f"Probabilités de fin d'alerte sauvegardées dans : {out_path}")


if __name__ == "__main__":
    main()

