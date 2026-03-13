"""
Clustering des types d'orages au niveau alerte (Data Battle 2026).

Objectif :
- Regrouper les alertes en quelques profils d'orages (petits / longs / intenses / diffus, etc.)
- Fournir une typologie simple à expliquer à l'oral.

Entrée : alerts_preprocessed.csv (généré par preprocessing_databattle_2026.py)
Sortie : affichage terminal (aucun fichier écrit)
"""

import pathlib

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
BASE_DIR = pathlib.Path(__file__).resolve().parent


def main() -> None:
    csv_path = BASE_DIR / "alerts_preprocessed.csv"
    if not csv_path.exists():
        print(f"Fichier manquant : {csv_path}")
        print("  Lancer d'abord : python3 preprocessing_databattle_2026.py")
        return

    df = pd.read_csv(csv_path)
    print(f"Chargement : {csv_path}  ({len(df)} alertes)")

    # Sélection de variables physiques pour décrire un orage
    candidate_cols = [
        "duration_total_minutes",   # durée de l'alerte
        "n_lightnings",            # activité totale
        "n_cloud_ground",          # éclairs au sol
        "n_intra_cloud",           # éclairs intra-nuage
        "mean_amplitude",          # intensité moyenne
        "mean_dist",               # distance moyenne
        "std_dist",                # dispersion spatiale radiale
        "mean_maxis",              # échelle / incertitude
        "max_maxis",
        "density",                 # n_lightnings / mean_maxis
        "storm_size_km",           # taille approximative de l'orage
    ]
    feature_cols = [c for c in candidate_cols if c in df.columns]
    if not feature_cols:
        print("Aucune des colonnes de features attendues n'est disponible.")
        return

    X = df[feature_cols].copy()

    # Remplacer d'éventuels NaN par les médianes
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Nombre de clusters : 4 profils d'orages (compromis lisibilité / finesse)
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    df["cluster"] = labels

    print(f"\nClustering KMeans avec k = {k} (types d'orages)")
    print("Répartition des alertes par cluster :")
    counts = df["cluster"].value_counts().sort_index()
    for cl, n in counts.items():
        pct = 100.0 * n / len(df)
        print(f"  Cluster {cl} : {n} alertes ({pct:.1f} %)")

    # Centres de clusters dans l'espace original (unités métier)
    centers_scaled = kmeans.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)
    centers_df = pd.DataFrame(centers, columns=feature_cols)

    print("\nCentres de clusters (médian typique par type d'orage) :")
    pd.set_option("display.max_columns", None)
    print(centers_df.round(2))

    # Résumé lisible par cluster
    print("\nRésumé interprétable par type d'orage (ordre croissant de durée):")
    centers_df["cluster"] = centers_df.index
    centers_sorted = centers_df.sort_values("duration_total_minutes")
    for _, row in centers_sorted.iterrows():
        cl = int(row["cluster"])
        dur = row["duration_total_minutes"]
        n_tot = row["n_lightnings"]
        n_cg = row["n_cloud_ground"]
        size_km = row.get("storm_size_km", np.nan)
        dens = row.get("density", np.nan)

        print(f"\n--- Cluster {cl} ---")
        print(f"  Durée typique           : ~{dur:.1f} minutes")
        print(f"  Nombre d'éclairs        : ~{n_tot:.0f} (dont ~{n_cg:.0f} au sol)")
        if not np.isnan(size_km):
            print(f"  Taille orage (storm_size_km) : ~{size_km:.1f} km")
        if not np.isnan(dens):
            print(f"  Densité (n_lightnings/mean_maxis) : ~{dens:.1f}")

        # Typologie simple
        comment = []
        if dur < 10 and n_tot < 20:
            comment.append("petit orage bref / local")
        if dur > 30 and n_tot > 50:
            comment.append("orage long et très actif")
        if not np.isnan(size_km) and size_km > 50:
            comment.append("système étendu (ligne / grand amas)")
        if not np.isnan(dens) and dens > 50:
            comment.append("forte densité d'éclairs (orage intense)")

        if comment:
            print("  Profil :", ", ".join(comment))

    # Sauvegarde pour utilisation dans le modèle (type d'orage comme feature)
    out_path = BASE_DIR / "alerts_preprocessed_with_cluster.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFichier avec cluster sauvegardé : {out_path}")
    print("  Utiliser ce fichier dans advanced_modeling pour inclure le type d'orage.")


if __name__ == "__main__":
    main()

