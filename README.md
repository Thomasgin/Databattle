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

## 💡 Solution proposée
Notre pipeline comporte trois étapes :
- **Clustering** (`clustering.py`) pour enrichir les alertes avec un type d'orage (`storm_type`).
- **Modélisation** (`modele.py`) pour comparer plusieurs modèles de régression (linéaire, Random Forest, XGBoost, CatBoost, MLP selon options).
- **Aide à la décision** (`probabilite_par_minute.py`) pour traduire les prédictions en minute de levée selon un seuil de confiance.

Le modèle final est sélectionné automatiquement selon la **MAE** la plus faible.

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
