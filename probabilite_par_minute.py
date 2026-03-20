"""
Data Battle 2026 – Grade 2 project.
Uses the BEST model (lowest MAE from advanced_modeling: RF/ET/GBR tuned).
Shows: aide à la décision (% → minute), puis comparaison avant/après modèle vs règle 30 min.
No files created; all output in terminal.
"""
import pathlib
from math import erf, sqrt

import numpy as np
import pandas as pd

MAX_MINUTE = 120


def normal_cdf(x: np.ndarray) -> np.ndarray:
    """CDF de la loi normale standard."""
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def run_probabilites(base_dir: pathlib.Path | None = None) -> None:
    """Affiche l'aide décision (confiance → minute) et le gain vs règle 30 min."""
    if base_dir is None:
        base_dir = pathlib.Path(__file__).resolve().parent
    preds_path = base_dir / "advanced_model_predictions.csv"
    if not preds_path.exists():
        preds_path = base_dir / "model_validation_predictions.csv"
        pred_col = "duration_pred"
    else:
        pred_col = "duration_pred_best"

    if not preds_path.exists():
        print("Fichier de prédictions absent. Exécuter d'abord : preprocessing puis advanced_modeling.")
        return

    print("Data Battle 2026 – Best model (from advanced_modeling).")
    print(f"Chargement des prédictions depuis : {preds_path}")
    df = pd.read_csv(preds_path)

    y_pred = df[pred_col].values
    if "duration_true" in df.columns:
        residuals = df["duration_true"].values - y_pred
        sigma = float(np.std(residuals))
    else:
        sigma = 27.0

    minutes = np.arange(0, MAX_MINUTE + 1, dtype=float)
    probs = normal_cdf((minutes - y_pred[:, np.newaxis]) / sigma)

    median_probs = np.median(probs, axis=0)
    print(f"Sigma (écart-type résidus) = {sigma:.2f} min.\n")
    print("=" * 60)
    print("  AIDE À LA DÉCISION – Seuil de confiance → minute de levée (médiane)")
    print("=" * 60)
    print("\n  Pour atteindre un niveau de confiance Y %, minute médiane pour lever l'alerte :")
    print("\n  Confidence %  |  Median minute (lift alert at this minute)")
    print("  --------------|------------------------------------------")
    for pct in [70, 75, 80, 85, 88, 90, 92, 94, 95, 96, 97, 98, 99]:
        thresh = pct / 100.0
        # Robustesse : on calcule la première minute pour chaque alerte,
        # puis on prend la médiane des minutes (au lieu de la minute sur la courbe médiane).
        # Ça évite les plateaux "même minute" quand la courbe médiane franchit tous les seuils
        # très tôt à cause d'une sigma (incertitude) trop optimiste.
        minute_per_alert = np.array(
            [np.searchsorted(probs[i, :], thresh) for i in range(len(probs))],
            dtype=int,
        )
        minute_per_alert = np.minimum(minute_per_alert, MAX_MINUTE)
        minute = int(np.median(minute_per_alert))
        print(f"       {pct:2d}%       |       {minute:3d} min")

    # Before model vs with model: average and median time, then improvement
    # Formulas (no change from original logic):
    #   BEFORE = current rule: lift 30 min after last lightning → time from start = duration_true + 30
    #   WITH MODEL = for each alert, first minute t where P(end before t) >= 95% (from y_pred + sigma)
    #   GAIN = (time before) - (time with model) → positive = we lift earlier with the model
    if "duration_true" in df.columns:
        duration_true = df["duration_true"].values
        temps_actuel = duration_true + 30  # current rule: lift at (storm end + 30 min) from start
        minute_at_95 = np.array([np.searchsorted(probs[i, :], 0.95) for i in range(len(probs))])
        minute_at_95 = np.minimum(minute_at_95, MAX_MINUTE)
        gain = temps_actuel - minute_at_95  # positive = we lift earlier with the model
        n_plus_tot = (gain > 0).sum()
        n_total = len(gain)

        mean_before = float(np.mean(temps_actuel))
        median_before = float(np.median(temps_actuel))
        mean_with_model = float(np.mean(minute_at_95))
        median_with_model = float(np.median(minute_at_95))
        gain_mean = float(np.mean(gain))
        gain_median = float(np.median(gain))

        # Sanity check: mean_before should equal mean(duration_true) + 30
        assert np.isclose(mean_before, float(np.mean(duration_true)) + 30.0), "Bug: temps_actuel formula"
        assert np.isclose(gain_mean, mean_before - mean_with_model), "Bug: gain formula"

        print("\n" + "=" * 60)
        print("  BEFORE THE MODEL (current rule: 30 min after last lightning)")
        print("=" * 60)
        print(f"  Average time to lift alert (from alert start): {mean_before:.1f} min")
        print(f"  Median  time to lift alert (from alert start): {median_before:.1f} min")

        print("\n" + "=" * 60)
        print("  WITH THE MODEL (95% confidence threshold)")
        print("=" * 60)
        print(f"  Average time to lift alert (from alert start): {mean_with_model:.1f} min")
        print(f"  Median  time to lift alert (from alert start): {median_with_model:.1f} min")

        print("\n" + "=" * 60)
        print("  MINUTES WON (before vs after model)")
        print("  Formula: gain = (time before) - (time with model) per alert.")
        print("=" * 60)
        print(f"  Gain (average): {gain_mean:+.1f} min  <- minutes we win on average with the model.")
        print(f"  Gain (median): {gain_median:+.1f} min")
        print(f"  In {n_plus_tot}/{n_total} alerts ({100*n_plus_tot/n_total:.0f}%) we lift earlier than with the 30-min rule.")
        if gain_median > 0:
            print("  --> IMPROVEMENT: the model allows lifting the alert earlier on average.")
        elif gain_median < 0:
            print("  --> NO IMPROVEMENT (median): the model waits longer than the current rule on average.")
        else:
            print("  --> Neutral: same median time.")


def main() -> None:
    run_probabilites()


if __name__ == "__main__":
    main()