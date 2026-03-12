# Data Battle 2026 – Projet Grade 2

Modèle probabiliste pour décider quand lever une alerte orage (comparaison avec la règle « 30 min après le dernier éclair »).

## Fichiers utiles au fonctionnement

- **`preprocessing_databattle_2026.py`** – Prétraitement des données brutes → `alerts_preprocessed.csv`
- **`advanced_modeling_databattle_2026.py`** – Entraînement du meilleur modèle (RF/ET/GBR) → prédictions
- **`probability_per_minute.py`** – Affichage en terminal : % par minute, comparaison avant/après modèle, minutes gagnées
- **`plot_probability_per_minute.py`** – Génère le graphique de décision **`graph_decision_minutes_vs_percentage.png`** (minutes vs confiance %, avec/sans modèle)

Données : `data_train_databattle2026/` (voir `info_data.md`). Le CSV brut est dans `.gitignore`.

## Lancer le projet

1. Mettre le fichier de données dans `data_train_databattle2026/segment_alerts_all_airports_train.csv`
2. `pip install -r requirements.txt`
3. `python3 preprocessing_databattle_2026.py`
4. `python3 advanced_modeling_databattle_2026.py`
5. `python3 probability_per_minute.py`  → résultats et gain en terminal
6. `python3 plot_probability_per_minute.py`  → génère le graphique

## Documentation

- `journal_projet_databattle_2026.md` – Journal du projet
- `presentation_sujet_lundi_2_mars.pdf` – Sujet Data Battle
