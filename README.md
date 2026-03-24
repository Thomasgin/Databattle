# 🏆 Data Battle IA PAU 2026 - Projet Prédiction durée d'alerte orage

## 👥 Équipe
- Nom de l’équipe : DBPower
- Membres :
  - Hugo Dury
  - Mathis Bohain
  - Thierry Berard
  - Imane Safe
  - Hichem Chergui
  - Thomas Gineste

## 🎯 Problématique
Nous cherchons à prédire la durée d'une alerte orage en minutes à partir de données météo et d'activité foudre.  
L'objectif est d'améliorer la décision de levée d'alerte par rapport à une règle fixe, avec une approche robuste (validation out-of-fold, GroupKFold).

Les données d'alerte sont fournies au niveau agrégé par alerte (une ligne = une alerte), avec une colonne **`airport`** pour identifier le site. Nous ne tranchons pas ici un rayon géographique en kilomètres : les documents du challenge peuvent différer ; le périmètre effectif est celui du jeu de données utilisé.

## 💡 Solution proposée
### Axe 1 — Fin d'alerte et aide à la décision
- **Clustering** (`clustering.py`) : typologie d'orages (`storm_type`).
- **Modélisation** (`modele.py`) : comparaison de modèles de régression ; sélection par **MAE** minimale.
- **Probabilités / seuils** (`probabilite_par_minute.py`) : traduction des prédictions en minutes de levée selon un niveau de confiance, et comparaison avec une règle-type « attente après dernier éclair ».

### Axe 2 — Tendances par aéroport
- **Analyse descriptive** (`analyse_par_aeroport.py`) : statistiques par `airport`, répartition des `storm_type` par site (fichiers CSV exportés pour le jury).

Le pipeline principal est `main.py` (clustering → modèle → analyse par aéroport → probabilités, sauf options `--skip-*`).

## ⚙️ Stack technique
- Langages :
  - Python 3
- Frameworks :
  - Scikit-learn
  - XGBoost (optionnel)
  - CatBoost (optionnel)
- Outils :
  - Pandas, NumPy
  - Matplotlib, Seaborn
  - Git / GitHub
- IA (si utilisé) :
  - Modèles de régression supervisée : `LinearRegression`, `RandomForestRegressor`, `XGBRegressor`, `CatBoostRegressor`, `MLPRegressor`

## 🚀 Installation & exécution

### Prérequis
- Python 3.10+
- `pip`
- Données CSV à la racine du projet (notamment `alerts_preprocessed.csv` ou `alerts_with_clusters.csv`)

### Installation
Depuis la racine du projet :

```bash
python3 -m pip install -r requirements.txt
```

### Exécution
Pipeline complet :

```bash
python3 main.py
```

Pipeline sans relancer le clustering :

```bash
python3 main.py --skip-clustering
```

Modèle seul sur CSV déjà clusterisé :

```bash
python3 modele.py --csv alerts_with_clusters.csv --xgboost on
```

Probabilités seules :

```bash
python3 probabilite_par_minute.py
```

Analyse par aéroport seule (après clustering) :

```bash
python3 analyse_par_aeroport.py --csv alerts_with_clusters.csv
```

Pipeline sans l'étape « tendances par aéroport » :

```bash
python3 main.py --skip-analyse-aeroport
```

À chaque run de `modele.py`, un fichier **`compute_footprint_estimate_simple.csv`** est créé : estimation **ordre de grandeur** kWh et kg CO₂eq (hypothèses 45 W, ~56 g/kWh — voir `ENVIRONNEMENT_SOCIAL.md` §2.3). Aucun outil obligatoire.

Optionnel — **CodeCarbon** (après `pip install codecarbon`) : `python3 modele.py ... --codecarbon` → `codecarbon_emissions_databattle.csv`.

## 📊 Livrables d'évaluation (jury)
- `model_benchmark_report.csv` : comparaison des modèles (MAE, RMSE, temps de calcul).
- `model_explainability_top_features.csv` : top variables explicatives du meilleur modèle (si disponible).
- `compute_footprint_proxy.csv` : proxy d'empreinte calcul (temps par modèle et part relative).
- `compute_footprint_estimate_simple.csv` : estimation indicative kWh / kg CO₂eq (temps mesuré × hypothèses ; §2.3 `ENVIRONNEMENT_SOCIAL.md`).
- `codecarbon_emissions_databattle.csv` : uniquement si `--codecarbon` + `pip install codecarbon`.
- `ENVIRONNEMENT_SOCIAL.md` : impacts, arbre de conséquences, méthode de quantification simple.
- `GOUVERNANCE_PROJET.md` : gouvernance, processus de décision et plan de poursuite.
- `tendances_par_aeroport.csv` : synthèse quantitative par aéroport (après exécution du pipeline ou de `analyse_par_aeroport.py`).
- `repartition_storm_type_par_aeroport.csv` : répartition des clusters par site (si `storm_type` présent).
