# Data Battle 2026 - Pipeline prédiction de durée d'alerte

Ce projet exécute une chaîne en 3 étapes :
1. **Clustering** des alertes météo (création de `storm_type`).
2. **Régression** de la durée d'alerte (`duration_total_minutes` ou `duration_minutes`).
3. **Aide à la décision** minute par minute (probabilité de fin d'alerte).

Le point d'entrée recommandé est `main.py`, qui orchestre ces 3 étapes.

---

## Objectif

Prédire la durée d'une alerte orageuse et comparer une levée basée modèle à la règle opérationnelle "30 minutes après le dernier éclair".

Sorties principales :
- `alerts_with_clusters.csv` : données enrichies avec `storm_type` (sortie clustering).
- `advanced_model_predictions.csv` : prédictions **out-of-fold** du meilleur modèle (sortie régression).
- sortie terminal de `probabilite_par_minute.py` : minute recommandée selon un seuil de confiance.

---

## Prérequis

Depuis la racine du projet :

```bash
pip install -r requirements.txt
```

Le `requirements.txt` inclut :
- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `seaborn`

> Remarque : si `xgboost` n'est pas installé, `modele.py` continue avec les autres modèles et affiche un message d'information.

---

## Structure des fichiers

| Fichier | Rôle |
|---|---|
| `main.py` | Pipeline principal : clustering -> modèle -> probabilités |
| `clustering.py` | Prétraitement + K-Means (4 clusters) + visualisations |
| `modele.py` | Entraînement/évaluation de plusieurs modèles de régression |
| `probabilite_par_minute.py` | Conversion des prédictions en probas et comparaison avec la règle 30 min |
| `alerts_preprocessed.csv` | Entrée attendue du clustering |
| `alerts_with_clusters.csv` | Sortie du clustering, entrée du modèle |
| `advanced_model_predictions.csv` | Prédictions finales out-of-fold du meilleur modèle |

---

## Détail du pipeline

### 1) Clustering (`clustering.py`)

Entrée : `alerts_preprocessed.csv` (ou fichier fourni via `--input` dans `main.py`).

Traitements effectués :
- feature engineering : `speed`, `storm_surface`
- encodage cyclique : `hour_sin`, `hour_cos`
- transformations logarithmiques : `log_n_lightnings`, `log_lightning_per_minute`, `log_storm_surface`
- clipping de certaines variables extrêmes (quantile 99%)
- standardisation puis `KMeans(n_clusters=4)`

Sorties :
- ajout de la colonne `storm_type`
- sauvegarde de `cluster_analysis.png`
- génération de `alerts_with_clusters.csv` (quand exécuté en script direct)

### 2) Modélisation (`modele.py`)

Colonne cible acceptée :
- `duration_total_minutes` (prioritaire), sinon `duration_minutes`.

Modèles évalués :
- `linear_baseline` : régression linéaire + standardisation
- `rf_default` : Random Forest (paramètres par défaut du script)
- `rf_tuned` : Random Forest + `RandomizedSearchCV`
- `xgb_tuned` : XGBoost + `RandomizedSearchCV` (si disponible)

Sélection finale : modèle avec **MAE minimale**.

Sortie :
- `advanced_model_predictions.csv` avec :
  - `duration_true`
  - `duration_pred_best`

### 3) Probabilités (`probabilite_par_minute.py`)

Le script lit `advanced_model_predictions.csv` (ou fallback `model_validation_predictions.csv`), estime l'incertitude via l'écart-type des résidus, puis :
- calcule \( P(\text{fin avant } t) \) pour `t = 0..120`
- affiche la minute médiane de levée pour des seuils 70% à 99%
- compare :
  - **avant modèle** : `duration_true + 30`
  - **avec modèle** : première minute où \(P \ge 95\%\)
- affiche le gain moyen/médian (minutes gagnées ou perdues)

---

## Évaluation et risque de fuite de données

Points robustes déjà en place :
- prédictions finales en **out-of-fold** (pas de `fit` sur tout le jeu puis `predict` sur ce même jeu)
- si `alert_airport_id` existe : validation en **GroupKFold** pour garder une alerte entière dans un seul pli
- pour `rf_tuned`/`xgb_tuned` : tuning fait à l'intérieur des plis d'entraînement (logique de nested CV)

Limite actuelle :
- le clustering est appris globalement avant la régression, ce qui peut introduire une fuite transductive via `storm_type`.

---

## Commandes d'exécution

### Pipeline complet (recommandé)

```bash
python3 main.py
```

### Pipeline sans relancer le clustering

```bash
python3 main.py --skip-clustering
```

Précondition : `alerts_with_clusters.csv` doit déjà exister.

### Pipeline sans étape probabilités

```bash
python3 main.py --skip-clustering --skip-probabilites
```

### Spécifier un CSV d'entrée pour le clustering

```bash
python3 main.py --input mon_fichier.csv
```

### Modèle seul

```bash
python3 modele.py --csv alerts_with_clusters.csv
```

Ou, sans `--csv`, le script cherche automatiquement (dans cet ordre) :
1. `alerts_with_clusters.csv`
2. si `--enriched` : `alerts_final_model_17var.csv`, puis `alerts_final_model_enriched.csv`
3. sinon : `alerts_final_model.csv`

### Probabilités seules

```bash
python3 probabilite_par_minute.py
```

---

## Erreurs fréquentes et solutions

- **`FileNotFoundError` sur `alerts_with_clusters.csv`**
  - lancer `python3 main.py` (sans `--skip-clustering`) ou exécuter `clustering.py` avant.

- **Dépendances graphiques manquantes (`matplotlib`/`seaborn`)**
  - installer via `pip install -r requirements.txt`
  - ou lancer `python3 main.py --skip-clustering` si `alerts_with_clusters.csv` existe déjà.

- **Colonne cible absente**
  - le CSV modèle doit contenir `duration_total_minutes` ou `duration_minutes`.

---

## Reproductibilité

- graine fixée à `42` dans les scripts principaux.
- validation croisée à 3 plis.
- exécution volontairement séquentielle (`n_jobs=1` côté CV/modèles) pour limiter les blocages CPU/RAM.
