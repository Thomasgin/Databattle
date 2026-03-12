"""
Data Battle 2026 – Decision graph: minutes vs confidence (%).
Use it to decide either by number of minutes OR by the risk/confidence level you accept.
Generates graph_decision_minutes_vs_percentage.png.
"""
import pathlib
from math import erf, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = pathlib.Path(__file__).resolve().parent
MAX_MINUTE = 120
OUT_PATH = BASE_DIR / "graph_decision_minutes_vs_percentage.png"


def normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def main() -> None:
    preds_path = BASE_DIR / "advanced_model_predictions.csv"
    if not preds_path.exists():
        preds_path = BASE_DIR / "model_validation_predictions.csv"
        pred_col = "duration_pred"
    else:
        pred_col = "duration_pred_best"

    if not preds_path.exists():
        print("Run preprocessing and advanced_modeling first.")
        return

    df = pd.read_csv(preds_path)
    y_pred = df[pred_col].values
    if "duration_true" in df.columns:
        duration_true = df["duration_true"].values
        sigma = float(np.std(duration_true - y_pred))
        temps_sans_modele = duration_true + 30
        median_sans = float(np.median(temps_sans_modele))
        mean_sans = float(np.mean(temps_sans_modele))
    else:
        sigma = 27.0
        median_sans = mean_sans = None

    minutes = np.arange(0, MAX_MINUTE + 1, dtype=float)
    probs = normal_cdf((minutes - y_pred[:, np.newaxis]) / sigma)
    median_probs = np.median(probs, axis=0)
    pct_curve = 100.0 * median_probs

    # Avec modèle (seuil 95%) : minute de levée par alerte
    minute_at_95 = np.array([np.searchsorted(probs[i, :], 0.95) for i in range(len(probs))])
    minute_at_95 = np.minimum(minute_at_95, MAX_MINUTE)
    median_avec = float(np.median(minute_at_95))
    mean_avec = float(np.mean(minute_at_95))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Courbe principale
    ax.plot(minutes, pct_curve, color="steelblue", linewidth=2.5, label="Confidence (%)")

    # Lignes de référence légères
    for pct in [90, 95, 99]:
        ax.axhline(y=pct, color="gray", linestyle=":", alpha=0.3, linewidth=0.7)

    # Un seul tableau : comparaison médiane / moyenne, sans modèle vs avec modèle (lisible, pas surchargé)
    table_data = [
        ["Sans modèle", f"{median_sans:.0f} min" if median_sans is not None else "—", f"{mean_sans:.0f} min" if mean_sans is not None else "—"],
        ["Avec modèle (95%)", f"{median_avec:.0f} min", f"{mean_avec:.0f} min"],
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=["", "Médiane", "Moyenne"],
        loc="lower right",
        cellLoc="center",
        colColours=["#f0f0f0", "#f0f0f0", "#f0f0f0"],
        bbox=[0.72, 0.18, 0.26, 0.22],
        fontsize=11,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight="bold")
        if i == 1:
            cell.set_facecolor("#fff0f0")
        if i == 2:
            cell.set_facecolor("#f0f8ff")

    ax.set_xlabel("Minutes since alert start", fontsize=12)
    ax.set_ylabel("Confidence (%)", fontsize=12)
    ax.set_title("Data Battle 2026 – Confidence vs time. Table: lift time without vs with model.", fontsize=11)
    ax.set_xlim(0, MAX_MINUTE)
    ax.set_ylim(0, 102)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_yticks([0, 20, 40, 60, 80, 90, 95, 100])
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Decision graph saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
