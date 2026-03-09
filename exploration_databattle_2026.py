import pathlib

import pandas as pd


def main() -> None:
    """Exploration initiale du dataset Data Battle 2026."""
    base_dir = pathlib.Path(__file__).resolve().parent
    data_path = base_dir / "data_train_databattle2026" / "segment_alerts_all_airports_train.csv"

    print(f"Chargement des données depuis : {data_path}")
    df = pd.read_csv(data_path)

    # Conversion de la colonne date en datetime si elle existe
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    print("\n=== Aperçu des données (5 premières lignes) ===")
    print(df.head())

    print("\n=== Informations générales ===")
    print(df.info())

    # Résumé statistique global (colonnes numériques)
    summary_path = base_dir / "summary_global.csv"
    df.describe(include="all").to_csv(summary_path)
    print(f"\nRésumé statistique global sauvegardé dans : {summary_path}")

    # Comptage par aéroport
    if "airport" in df.columns:
        airport_counts = df["airport"].value_counts().rename_axis("airport").reset_index(name="n_lightnings")
        airport_counts_path = base_dir / "summary_airport_counts.csv"
        airport_counts.to_csv(airport_counts_path, index=False)
        print(f"Comptage des éclairs par aéroport sauvegardé dans : {airport_counts_path}")

    # Analyse des alertes si disponible
    alert_cols = {"alert_airport_id", "is_last_lightning_cloud_ground"}
    if alert_cols.issubset(df.columns):
        # Nombre de lignes avec une alerte identifiée
        n_alert_rows = df["alert_airport_id"].notna().sum()
        print(f"\nNombre de lignes avec alert_airport_id non nul : {n_alert_rows}")

        # Résumé par alerte (taille de l'alerte)
        alerts_grouped = (
            df.loc[df["alert_airport_id"].notna(), ["airport", "alert_airport_id", "date", "is_last_lightning_cloud_ground"]]
            .groupby(["airport", "alert_airport_id"], as_index=False)
            .agg(
                n_lightnings=("date", "size"),
                start_time=("date", "min"),
                end_time=("date", "max"),
                n_last_cloud_ground=("is_last_lightning_cloud_ground", lambda s: s.fillna(False).sum()),
            )
        )

        alerts_grouped["duration_minutes"] = (
            (alerts_grouped["end_time"] - alerts_grouped["start_time"]).dt.total_seconds() / 60.0
        )

        alerts_summary_path = base_dir / "summary_alerts.csv"
        alerts_grouped.to_csv(alerts_summary_path, index=False)
        print(f"Résumé des alertes sauvegardé dans : {alerts_summary_path}")
    else:
        print("\nColonnes d'alerte manquantes, aucune analyse d'alerte effectuée.")


if __name__ == "__main__":
    main()

