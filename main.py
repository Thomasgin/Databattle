import argparse
import pathlib

from modele import run_model
from probabilite_par_minute import run_probabilites


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline: clustering.py -> modele.py")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="CSV d'entrée pour le clustering (défaut: alerts_preprocessed.csv).",
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Ne relance pas le clustering; utilise alerts_with_clusters.csv existant.",
    )
    parser.add_argument(
        "--skip-probabilites",
        action="store_true",
        help="N'exécute pas probabilite_par_minute.py après le modèle.",
    )
    parser.add_argument(
        "--xgboost",
        choices=["ask", "on", "off"],
        default="ask",
        help="Active XGBoost pour le modèle: ask (demande), on (force), off (désactive).",
    )
    parser.add_argument(
        "--skip-analyse-aeroport",
        action="store_true",
        help="N'exécute pas l'analyse des tendances par aéroport (fichiers CSV).",
    )
    parser.add_argument(
        "--codecarbon",
        action="store_true",
        help="Passe l'option à modele.py : mesure CodeCarbon (énergie / CO₂eq) pendant le run modèle.",
    )
    args = parser.parse_args()

    base_dir = pathlib.Path(__file__).resolve().parent
    input_csv = pathlib.Path(args.input) if args.input else (base_dir / "alerts_preprocessed.csv")
    out_csv = base_dir / "alerts_with_clusters.csv"

    if not args.skip_clustering:
        try:
            # Import local ici pour éviter la dépendance seaborn/matplotlib
            # quand on utilise seulement un CSV déjà clusterisé.
            from clustering import load_and_clean, run_clustering
        except ModuleNotFoundError as exc:
            if exc.name in {"seaborn", "matplotlib"}:
                if out_csv.exists():
                    print(
                        "[1/3] Dépendance manquante pour clustering "
                        f"({exc.name}). Fallback vers CSV existant: {out_csv.name}"
                    )
                else:
                    raise ModuleNotFoundError(
                        f"Dépendance manquante: {exc.name}. "
                        "Installer avec: pip install seaborn matplotlib "
                        "ou relancer avec --skip-clustering si le CSV clusterisé existe."
                    ) from exc
            else:
                raise
        else:
            if not input_csv.exists():
                raise FileNotFoundError(f"Fichier d'entrée introuvable pour le clustering: {input_csv}")
            print(f"[1/3] Clustering sur: {input_csv}")
            df = load_and_clean(str(input_csv))
            run_clustering(df)
    else:
        print("[1/3] Clustering ignoré (--skip-clustering).")

    if not out_csv.exists():
        raise FileNotFoundError(
            f"Fichier attendu non trouvé: {out_csv}. "
            "Exécute d'abord clustering.py ou main.py sans --skip-clustering."
        )

    print(f"[2/3] Modèle de régression sur: {out_csv}")
    run_model(
        csv_path=str(out_csv),
        use_enriched=False,
        xgboost_mode=args.xgboost,
        use_codecarbon=args.codecarbon,
    )

    if not args.skip_analyse_aeroport:
        from analyse_par_aeroport import run_analyse_par_aeroport

        print("[2b/3] Analyse tendances par aéroport...")
        run_analyse_par_aeroport(base_dir, csv_path=str(out_csv))

    if not args.skip_probabilites:
        print("[3/3] Probabilités par minute (terminal)...")
        run_probabilites(base_dir)
    else:
        print("[3/3] Probabilités ignorées (--skip-probabilites).")


if __name__ == "__main__":
    main()
