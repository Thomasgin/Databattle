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
    required_cols = {"airport", "alert_airport_id", "date", "icloud"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la construction des alertes : {missing}")

    # On ne conserve que les lignes qui appartiennent à une alerte identifiée
    alerts_df = df[df["alert_airport_id"].notna()].copy()
    print(f"Nombre de lignes appartenant à une alerte identifiée : {len(alerts_df)}")

    # maxis : présent dans le brut (erreur de localisation en km) ; on l'agrège pour caractériser l'orage
    has_maxis = "maxis" in alerts_df.columns
    agg_dict = {
        "n_lightnings": ("date", "size"),
        "start_time": ("date", "min"),
        "end_time": ("date", "max"),
        "n_cloud_ground": ("icloud", lambda s: (~s.astype(bool)).sum()),
        "n_intra_cloud": ("icloud", lambda s: s.astype(bool).sum()),
        "last_cloud_ground_time": ("date", "max"),
        "mean_dist": ("dist", "mean"),
        "std_dist": ("dist", "std"),
        "mean_amplitude": ("amplitude", "mean"),
    }
    if "lon" in alerts_df.columns and "lat" in alerts_df.columns:
        agg_dict["min_lon"] = ("lon", "min")
        agg_dict["max_lon"] = ("lon", "max")
        agg_dict["min_lat"] = ("lat", "min")
        agg_dict["max_lat"] = ("lat", "max")
        agg_dict["mean_lat"] = ("lat", "mean")
    if has_maxis:
        agg_dict["mean_maxis"] = ("maxis", "mean")
        agg_dict["max_maxis"] = ("maxis", "max")

    grouped = (
        alerts_df.groupby(["airport", "alert_airport_id"], as_index=False)
        .agg(**agg_dict)
    )

    # Durées en minutes
    grouped["duration_total_minutes"] = (
        (grouped["end_time"] - grouped["start_time"]).dt.total_seconds() / 60.0
    )
    grouped["duration_until_last_cg_minutes"] = (
        (grouped["last_cloud_ground_time"] - grouped["start_time"]).dt.total_seconds() / 60.0
    )

    # std_dist à 0 si NaN (1 seul éclair). Pas de lightning_rate (éviter fuite: durée = cible)
    grouped["std_dist"] = grouped["std_dist"].fillna(0.0)

    # maxis : taille/incertitude orage (doc = erreur de localisation en km). Densité = intensité relative
    if has_maxis:
        grouped["mean_maxis"] = grouped["mean_maxis"].fillna(0.0)
        grouped["max_maxis"] = grouped["max_maxis"].fillna(0.0)
        grouped["density"] = grouped["n_lightnings"] / (grouped["mean_maxis"] + 0.01)

    # Dernier éclair de l'alerte : CG ou IC ? (phase active si dernier = CG)
    idx_last = alerts_df.groupby(["airport", "alert_airport_id"])["date"].idxmax()
    last_icloud = alerts_df.loc[idx_last, ["airport", "alert_airport_id", "icloud"]].copy()
    last_icloud["last_lightning_is_cg"] = (~last_icloud["icloud"].astype(bool)).astype(int)
    last_icloud = last_icloud[["airport", "alert_airport_id", "last_lightning_is_cg"]]
    grouped = grouped.merge(last_icloud, on=["airport", "alert_airport_id"], how="left")
    grouped["last_lightning_is_cg"] = grouped["last_lightning_is_cg"].fillna(0).astype(int)

    # Taille de l'orage (grand axe approximatif en km à partir de la boîte lon/lat)
    if "min_lon" in grouped.columns:
        lat_rad = np.radians(grouped["mean_lat"].values)
        dlat_km = (grouped["max_lat"] - grouped["min_lat"]).values * 111.0
        dlon_km = (grouped["max_lon"] - grouped["min_lon"]).values * 111.0 * np.cos(lat_rad)
        grouped["storm_size_km"] = np.sqrt(dlat_km**2 + dlon_km**2)
        grouped.drop(columns=["min_lon", "max_lon", "min_lat", "max_lat", "mean_lat"], inplace=True)

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

