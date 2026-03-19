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

Sinon, utilise `main.py` avec `--skip-clustering` (voir ci-dessous) si le fichier `alerts_with_clusters.csv` est déjà présent.

---

## Fichiers importants

| Fichier | Rôle |
|---------|------|
| `alerts_preprocessed.csv` | Entrée du clustering (une ligne = une alerte). |
| `alerts_with_clusters.csv` | Sortie du clustering (alertes + colonne `storm_type`). |
| `clustering.py` | Nettoyage + K-Means + visualisations (ne pas modifier si imposé). |
| `modele.py` | Random Forest (défaut + tuné) et XGBoost si installé. |
| `main.py` | Enchaîne clustering puis modèle. |

---

## Exécution recommandée (CSV déjà clusterisé)

La plus simple si `alerts_with_clusters.csv` existe déjà :

```bash
cd /chemin/vers/Databattle
python3 main.py --skip-clustering
```

---

## Exécution complète (clustering + modèle)

1. Avoir `alerts_preprocessed.csv` à la racine du projet.
2. Installer `seaborn` et `matplotlib` (voir prérequis).
3. Lancer :

```bash
cd /chemin/vers/Databattle
python3 main.py
```

Option : autre fichier d’entrée pour le clustering :

```bash
python3 main.py --input /chemin/vers/mon_fichier.csv
```

*(Le script attend en sortie `alerts_with_clusters.csv` à la racine du projet.)*

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

---

## Résolution de problèmes

- **`ModuleNotFoundError: seaborn`** : `pip install seaborn matplotlib` **ou** `python3 main.py --skip-clustering` si `alerts_with_clusters.csv` existe.
- **`alerts_with_clusters.csv` introuvable** : lancer d’abord le clustering (`python3 main.py` sans `--skip-clustering`) ou régénérer ce fichier.
