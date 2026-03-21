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
| `modele.py` | Baseline régression linéaire (OLS + scaling), Random Forest (défaut + tuné), XGBoost si installé ; génère `advanced_model_predictions.csv`. |
| `probabilite_par_minute.py` | Probabilités minute par minute + gain vs règle 30 min (lit `advanced_model_predictions.csv`). |
| `main.py` | Enchaîne : clustering → modèle → probabilités (étapes 1/3, 2/3, 3/3). |

### Évaluation sans fuite (important)

- **`advanced_model_predictions.csv`** contient des prédictions **out-of-fold** (chaque ligne est prédite par un modèle **n’ayant pas** vu cette ligne à l’entraînement). On ne fait plus `fit` sur tout le jeu puis `predict` sur le même jeu.
- Si la colonne **`alert_airport_id`** est présente, la validation utilise un **`GroupKFold`** : toutes les lignes d’une même alerte restent dans le même pli (évite la fuite entre lignes corrélées).
- Pour **RF tuné** et **XGB tuné**, le tuning est fait **à l’intérieur de chaque pli train** (outer GroupKFold + inner `RandomizedSearchCV` avec `GroupKFold` sur les mêmes groupes), pas sur tout le dataset puis CV affichée.

*Le clustering global (`clustering.py`) reste appris sur tout le jeu avant la régression : fuite transductive résiduelle possible sur `storm_type` ; une étape suivante serait un KMeans par pli ou sur train seulement.*

---

## Ouverture du projet — LSTM et données séquentielles

Aujourd’hui, `modele.py` travaille sur des **vecteurs tabulaires** (une ligne = alerte ou segment avec des **agrégats** : moyennes, ratios, etc.). Les modèles comparés sont donc adaptés à ce format : **régression linéaire**, **Random Forest**, **XGBoost** (éventuellement plus tard **MLP** sur le même vecteur).

**LSTM** (*Long Short-Term Memory*) est pensé pour des **séquences ordonnées** (dépendance au **temps** ou à l’**ordre des événements**), pas pour un simple vecteur de features par ligne.

**Piste future crédible :** reconstruire, pour chaque `alert_airport_id`, une **séquence** au fil du temps — par exemple une ligne par **minute** ou par **éclair** (débit, distance, type CG/IC, etc.) — puis :

1. **Aligner / padding** des séquences (longueurs variables) ;
2. Entraîner un **LSTM** (ou GRU) pour prédire la **durée restante** ou la **fin d’alerte** à partir du début (ou au fil de l’eau) ;
3. Garder le **même protocole d’évaluation** (split par groupe = alerte, pas de fuite temporelle si on prédit le futur à partir du passé).

Cela constitue un **autre paradigme** (séquentiel vs tabulaire) et une **vraie justification** d’utiliser un LSTM, plutôt que de l’appliquer à plat sur les colonnes actuelles.

*Prochaines étapes envisagées côté modèles : ajout d’un **MLP** sur les features actuelles pour une 4ᵉ famille « réseau de neurones dense », puis des **graphiques** de comparaison des métriques / résidus.*

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
