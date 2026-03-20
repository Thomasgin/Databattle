# Data Battle – Clustering + modèle de régression

## Prérequis

Dans le dossier du projet :

```bash
pip install -r requirements.txt
```

Pour **relancer le clustering** depuis `main.py` (graphiques + K-Means), installe aussi :

```bash
pip install seaborn matplotlib
```

## Option : activer XGBoost

Si tu veux que `modele.py` teste aussi `xgb_tuned` :

Dans un environnement virtuel (recommandé) :
```bash
source .venv/bin/activate
pip install xgboost
```

Sinon, `xgboost` n’est pas requis : le script affichera
`(XGBoost non installé : pip install xgboost)` et passera au reste.

Sinon, utilise `main.py` avec `--skip-clustering` (voir ci-dessous) si le fichier `alerts_with_clusters.csv` est déjà présent.

---

## Fichiers importants

| Fichier | Rôle |
|---------|------|
| `alerts_preprocessed.csv` | Entrée du clustering (une ligne = une alerte). |
| `alerts_with_clusters.csv` | Sortie du clustering (alertes + colonne `storm_type`). |
| `clustering.py` | Nettoyage + K-Means + visualisations (ne pas modifier si imposé). |
| `modele.py` | Random Forest (défaut + tuné) et XGBoost si installé ; génère `advanced_model_predictions.csv`. |
| `probabilite_par_minute.py` | Probabilités minute par minute + gain vs règle 30 min (lit `advanced_model_predictions.csv`). |
| `main.py` | Enchaîne : clustering → modèle → probabilités (étapes 1/3, 2/3, 3/3). |

---

## Exécution recommandée (CSV déjà clusterisé)

La plus simple si `alerts_with_clusters.csv` existe déjà :

```bash
cd /chemin/vers/Databattle
python3 main.py --skip-clustering
```

Pour **ne pas** lancer l’affichage des probabilités après le modèle :

```bash
python3 main.py --skip-clustering --skip-probabilites
```

---

## Exécution complète (clustering + modèle + probabilités)

1. Avoir `alerts_preprocessed.csv` à la racine du projet.
2. Installer `seaborn` et `matplotlib` (voir prérequis).
3. Lancer :

```bash
cd /chemin/vers/Databattle
python3 main.py
```

À la fin, le modèle écrit `advanced_model_predictions.csv`, puis `main.py` appelle `probabilite_par_minute.py` (sortie terminal uniquement).

---

## Probabilités seules (si le modèle a déjà tourné)

```bash
python3 probabilite_par_minute.py
```

---

## Modèle seul (sans passer par `main.py`)

Avec le CSV des clusters :

```bash
python3 modele.py --csv alerts_with_clusters.csv
```

Sans `--csv` : le script cherche d’abord `alerts_with_clusters.csv`, puis les autres CSV du projet selon la config.

Avec les données enrichies (si ces fichiers existent) :

```bash
python3 modele.py --enriched
```

Ensuite, pour les probabilités :

```bash
python3 probabilite_par_minute.py
```

---

## Résolution de problèmes

- **`ModuleNotFoundError: seaborn`** : `pip install seaborn matplotlib` **ou** `python3 main.py --skip-clustering` si `alerts_with_clusters.csv` existe.
- **`alerts_with_clusters.csv` introuvable** : lancer d’abord le clustering (`python3 main.py` sans `--skip-clustering`) ou régénérer ce fichier.
