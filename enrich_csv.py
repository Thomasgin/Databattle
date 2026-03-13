"""
Enrichit alerts_final_model.csv avec de nouvelles variables dérivées (sans fuite de données).
Sauvegarde dans alerts_final_model_enriched.csv.
Nouvelles variables :
  - cg_ratio : proportion éclairs nuage-sol (danger direct)
  - log_n_lightnings : log(1 + n_lightnings) pour réduire la skew
  - amplitude_range : max_amplitude - mean_amplitude (pic d'intensité)
  - proximity_ratio : min_dist / mean_dist (proximité du plus proche éclair)
  - hour_sin, hour_cos : encodage cyclique de l'heure (23h proche de 0h)
  - month_sin, month_cos : encodage cyclique du mois
  - severity_proxy : max_amplitude * log(1 + n_cloud_ground) (interaction danger)
"""
import pathlib
import numpy as np
import pandas as pd

BASE_DIR = pathlib.Path(__file__).resolve().parent
IN_PATH = BASE_DIR / "alerts_final_model.csv"
OUT_PATH = BASE_DIR / "alerts_final_model_enriched.csv"


def main():
    if not IN_PATH.exists():
        print(f"Fichier absent : {IN_PATH}")
        return

    df = pd.read_csv(IN_PATH)

    # Proportion nuage-sol (éviter division par zéro)
    n = df["n_lightnings"].values
    n_cg = df["n_cloud_ground"].values
    df["cg_ratio"] = np.where(n > 0, n_cg / n, 0.0)

    # Log du nombre d'éclairs (réduit l'effet des très grosses alertes)
    df["log_n_lightnings"] = np.log1p(n)

    # Écart d'amplitude (pic vs moyenne)
    df["amplitude_range"] = df["max_amplitude"] - df["mean_amplitude"]

    # Proximité : plus min_dist est petit par rapport à mean_dist, plus l'orage est "sur" l'aéroport
    mean_d = df["mean_dist"].values
    min_d = df["min_dist"].values
    df["proximity_ratio"] = min_d / (mean_d + 0.01)

    # Cyclique : heure (0-24)
    h = df["start_hour"].values
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)

    # Cyclique : mois (1-12)
    m = df["start_month"].values
    df["month_sin"] = np.sin(2 * np.pi * (m - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (m - 1) / 12)

    # Interaction : sévérité (pic d'amplitude × activité nuage-sol)
    df["severity_proxy"] = df["max_amplitude"] * np.log1p(np.maximum(n_cg, 0))

    # Jour de l'année cyclique
    d = df["start_dayofyear"].values
    df["day_sin"] = np.sin(2 * np.pi * (d - 1) / 365)
    df["day_cos"] = np.cos(2 * np.pi * (d - 1) / 365)

    # Dispersion distance (mouvement orage)
    df["dist_spread"] = np.where(mean_d > 0.01, df["std_dist"].values / (mean_d + 0.01), 0.0)

    # Activité × type (nombre éclairs × ratio intra-nuage)
    df["activity_ic"] = np.log1p(n) * df["ic_ratio"]

    # Danger proche : min_dist petit + beaucoup de nuage-sol
    df["danger_proxy"] = np.log1p(np.maximum(n_cg, 0)) / (min_d + 1.0)

    # Azimuth spread (orage étalé ou directionnel)
    df["azimuth_spread"] = np.log1p(np.maximum(df["std_azimuth"].values, 0))

    # Amplitude moyenne pondérée par activité
    df["amp_activity"] = df["mean_amplitude"] * np.log1p(n)

    df.to_csv(OUT_PATH, index=False)
    new_cols = [c for c in df.columns if c not in pd.read_csv(IN_PATH).columns]
    print(f"Enrichi : {IN_PATH} -> {OUT_PATH}")
    print(f"Nouvelles variables ({len(new_cols)}) : {new_cols}")

    # Version 17 variables (sous-ensemble optimal RFECV – meilleur compromis)
    COLS_17 = [
        "duration_total_minutes",
        "n_lightnings", "n_cloud_ground", "n_intra_cloud",
        "mean_dist", "std_dist",
        "mean_azimuth_sin", "mean_azimuth_cos", "std_azimuth",
        "start_hour", "start_dayofyear",
        "cg_ratio", "log_n_lightnings", "amplitude_range", "severity_proxy",
        "day_sin", "danger_proxy", "amp_activity",
    ]
    out_17 = BASE_DIR / "alerts_final_model_17var.csv"
    df_17 = df[[c for c in COLS_17 if c in df.columns]].copy()
    df_17.to_csv(out_17, index=False)
    print(f"Version 17 variables : {out_17}")


if __name__ == "__main__":
    main()
