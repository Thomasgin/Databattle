import pathlib
from typing import Optional

import numpy as np
import pandas as pd


def load_raw_data(base_dir: pathlib.Path) -> pd.DataFrame:
    """Charge le fichier brut et applique les conversions de base."""
    data_path = base_dir / "data_train_databattle2026" / "segment_alerts_all_airports_train.csv"
    print(f"Chargement des données brutes depuis : {data_path}")
    df = pd.read_csv(data_path)

    # Conversion de la date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    # Normalisation du nom de colonne d'alerte par cohérence avec la doc
    if "airport_alert_id" in df.columns and "alert_airport_id" not in df.columns:
        df = df.rename(columns={"airport_alert_id": "alert_airport_id"})
        print("Renommage de la colonne 'airport_alert_id' en 'alert_airport_id'.")

    return df


def build_alert_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """Construit un tableau agrégé au niveau de l'alerte."""
    required_cols = {"airport", "alert_airport_id", "date", "icloud", "is_last_lightning_cloud_ground"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la construction des alertes : {missing}")

    # On ne conserve que les lignes qui appartiennent à une alerte identifiée
    alerts_df = df[df["alert_airport_id"].notna()].copy()
    print(f"Nombre de lignes appartenant à une alerte identifiée : {len(alerts_df)}")

    grouped = (
        alerts_df.groupby(["airport", "alert_airport_id"], as_index=False)
        .agg(
            n_lightnings=("date", "size"),
            start_time=("date", "min"),
            end_time=("date", "max"),
            n_cloud_ground=("icloud", lambda s: (~s.astype(bool)).sum()),
            n_intra_cloud=("icloud", lambda s: s.astype(bool).sum()),
            last_cloud_ground_time=("date", "max"),
        )
    )

    # Durées en minutes
    grouped["duration_total_minutes"] = (
        (grouped["end_time"] - grouped["start_time"]).dt.total_seconds() / 60.0
    )
    grouped["duration_until_last_cg_minutes"] = (
        (grouped["last_cloud_ground_time"] - grouped["start_time"]).dt.total_seconds() / 60.0
    )

    # On ajoute des features calendaires simples basées sur le début d'alerte
    grouped["start_year"] = grouped["start_time"].dt.year
    grouped["start_month"] = grouped["start_time"].dt.month
    grouped["start_dayofyear"] = grouped["start_time"].dt.dayofyear
    grouped["start_hour"] = grouped["start_time"].dt.hour

    return grouped


def main() -> None:
    """Prétraitement des données Data Battle 2026 (niveau alerte)."""
    base_dir = pathlib.Path(__file__).resolve().parent

    df_raw = load_raw_data(base_dir)
    print("Données brutes chargées.")

    alerts_table = build_alert_level_table(df_raw)
    print("Tableau au niveau alerte construit.")

    output_path = base_dir / "alerts_preprocessed.csv"
    alerts_table.to_csv(output_path, index=False)
    print(f"Tableau d'alertes prétraité sauvegardé dans : {output_path}")

    # Petit résumé texte pour vérifier les ordres de grandeur
    n_alerts = len(alerts_table)
    n_airports = alerts_table["airport"].nunique()
    print(f"Nombre d'alertes distinctes : {n_alerts}")
    print(f"Nombre d'aéroports : {n_airports}")
    print("Durée totale (min) – min / médiane / max :")
    print(alerts_table["duration_total_minutes"].describe()[["min", "50%", "max"]])


if __name__ == "__main__":
    main()

