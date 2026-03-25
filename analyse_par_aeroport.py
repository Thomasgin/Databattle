"""
Tendances par aéroport (axe complémentaire du sujet Data Battle).

Agrège les alertes par la colonne `airport` présente dans les CSV fournis.
Ne fixe aucun rayon en km : le périmètre spatial est celui des données sources
(Météorage / documentation du jeu de données).

Sorties :
- tendances_par_aeroport.csv : statistiques par site
- repartition_storm_type_par_aeroport.csv : part de chaque cluster par aéroport
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

TARGET_CANDIDATES = ("duration_total_minutes", "duration_minutes")


def _resolve_csv(base_dir: pathlib.Path, csv_path: str | None) -> pathlib.Path:
    if csv_path:
        return pathlib.Path(csv_path)
    p = base_dir / "alerts_with_clusters.csv"
    if p.exists():
        return p
    return base_dir / "alerts_preprocessed.csv"


def run_analyse_par_aeroport(
    base_dir: pathlib.Path | None = None,
    csv_path: str | None = None,
) -> None:
    if base_dir is None:
        base_dir = pathlib.Path(__file__).resolve().parent
    path = _resolve_csv(base_dir, csv_path)
    if not path.exists():
        print(f"[analyse aéroport] Fichier introuvable : {path}")
        return

    df = pd.read_csv(path)
    if "airport" not in df.columns:
        print("[analyse aéroport] Colonne 'airport' absente, analyse ignorée.")
        return

    dur_col = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if dur_col is None:
        print("[analyse aéroport] Aucune colonne durée trouvée, analyse ignorée.")
        return

    exclude_num = {dur_col, "storm_type", "alert_airport_id"}
    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_num and not str(c).startswith("Unnamed")
    ][:12]

    rows = []
    for airport, g in df.groupby("airport", dropna=False):
        row = {
            "airport": airport,
            "n_alertes": len(g),
            f"{dur_col}_mediane": float(g[dur_col].median()),
            f"{dur_col}_moyenne": float(g[dur_col].mean()),
        }
        for c in numeric_cols:
            if c in g.columns:
                row[f"{c}_moyenne"] = float(g[c].mean())
        rows.append(row)

    out_main = base_dir / "tendances_par_aeroport.csv"
    pd.DataFrame(rows).sort_values("n_alertes", ascending=False).to_csv(
        out_main, index=False
    )
    print(f"[analyse aéroport] Sauvegardé : {out_main.name}")

    if "storm_type" in df.columns:
        ct = pd.crosstab(df["airport"], df["storm_type"], normalize="index") * 100.0
        ct = ct.round(2)
        out_st = base_dir / "repartition_storm_type_par_aeroport.csv"
        ct.to_csv(out_st)
        print(f"[analyse aéroport] Sauvegardé : {out_st.name}")

    print("\n  Résumé par aéroport (aperçu) :")
    summary = pd.DataFrame(rows)[
        ["airport", "n_alertes", f"{dur_col}_mediane", f"{dur_col}_moyenne"]
    ].sort_values("n_alertes", ascending=False)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse des tendances d'alertes par aéroport."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="CSV (défaut: alerts_with_clusters.csv ou alerts_preprocessed.csv).",
    )
    args = parser.parse_args()
    run_analyse_par_aeroport(csv_path=args.csv)


if __name__ == "__main__":
    main()
