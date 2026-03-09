import pathlib

import pandas as pd


def main() -> None:
    """Analyse descriptive des alertes par aéroport."""
    base_dir = pathlib.Path(__file__).resolve().parent
    alerts_path = base_dir / "alerts_preprocessed.csv"

    print(f"Chargement des alertes prétraitées depuis : {alerts_path}")
    df = pd.read_csv(alerts_path, parse_dates=["start_time", "end_time", "last_cloud_ground_time"])

    # Synthèse par aéroport
    summary = (
        df.groupby("airport")
        .agg(
            n_alerts=("alert_airport_id", "nunique"),
            mean_duration=("duration_total_minutes", "mean"),
            median_duration=("duration_total_minutes", "median"),
            max_duration=("duration_total_minutes", "max"),
            mean_n_lightnings=("n_lightnings", "mean"),
        )
        .reset_index()
    )

    summary_path = base_dir / "analysis_airport_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Résumé par aéroport sauvegardé dans : {summary_path}")

    # Saisonniété simple : nombre d'alertes par aéroport et par mois
    df["start_month"] = pd.to_datetime(df["start_time"]).dt.month
    monthly = (
        df.groupby(["airport", "start_month"])
        .agg(
            n_alerts=("alert_airport_id", "nunique"),
            mean_duration=("duration_total_minutes", "mean"),
        )
        .reset_index()
    )

    monthly_path = base_dir / "analysis_airport_monthly.csv"
    monthly.to_csv(monthly_path, index=False)
    print(f"Résumé par aéroport et par mois sauvegardé dans : {monthly_path}")


if __name__ == "__main__":
    main()

