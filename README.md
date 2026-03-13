# Data Battle 2026 – Modèle probabiliste de fin d’orage

Projet pour **prédire la durée d’une alerte orage** et en déduire une **probabilité de fin d’orage** minute par minute, afin d’aider à lever l’alerte plus tôt que la règle fixe (ex. 30 min après le dernier éclair). Un **clustering K-Means** identifie des types d’orages pour l’analyse et peut être utilisé comme entrée du modèle.

---

## 1. Objectif

- **Entrée :** données Météorage (éclairs dans un rayon ~20–50 km autour de plusieurs aéroports).
- **Sortie :** pour chaque alerte, (1) une prédiction de la durée totale en minutes, (2) une probabilité « fin d’orage avant t minutes » pour chaque minute, (3) la minute à laquelle un seuil de confiance (ex. 95 %) est atteint, et (4) une comparaison avec la règle actuelle (gain en minutes).
- **Clustering :** regrouper les alertes en 4 types d’orages (brefs, moyens, longs, très actifs) pour l’analyse et, optionnellement, comme variable d’entrée du modèle.

---

## 2. Données et pipeline

- **Brut :** `data_train_databattle2026/segment_alerts_all_airports_train.csv` — une ligne = un éclair (date, lon, lat, amplitude, maxis, icloud, dist, azimuth, airport, alert_airport_id, etc.).
- **Prétraitement :** `preprocessing_databattle_2026.py` agrège au **niveau alerte** (une ligne = une alerte) et produit `alerts_preprocessed.csv`.
- **Clustering :** `clustering_storm_types.py` lit ce CSV, applique K-Means (k=4), affiche les profils et sauvegarde `alerts_preprocessed_with_cluster.csv` (même table + colonne `cluster`).
- **Modèle :** `advanced_modeling_databattle_2026.py` charge de préférence le CSV avec cluster, entraîne un modèle de régression, sauvegarde les prédictions.
- **Probabilités et décision :** `probability_per_minute.py` (terminal) et `plot_probability_per_minute.py` (graphique) utilisent ces prédictions pour les probas par minute et la comparaison avant/après modèle.

---

## 3. Variables d’entrée du modèle (et leur rôle)

Le modèle prédit **duration_total_minutes** (durée totale de l’alerte en minutes). Toutes les variables ci-dessous sont calculées ou agrégées **par alerte** (sans utiliser la durée future, pas de fuite de données).

### 3.1 Variables d’activité électrique

| Variable | Rôle | Pourquoi on la prend |
|--------|------|----------------------|
| **n_lightnings** | Nombre total d’éclairs de l’alerte | Plus il y a d’éclairs, plus l’orage est actif et souvent plus long. |
| **n_cloud_ground** | Nombre d’éclairs nuage–sol (CG) | Les CG sont les plus dangereux ; proportion importante souvent liée à des orages plus intenses. |
| **n_intra_cloud** | Nombre d’éclairs intra-nuage (IC) | Complète le tableau activité / type d’orage. |
| **mean_amplitude** | Amplitude moyenne des éclairs | Intensité moyenne du phénomène ; orages plus intenses peuvent avoir des dynamiques différentes. |

### 3.2 Variables de position et de géométrie

| Variable | Rôle | Pourquoi on la prend |
|--------|------|----------------------|
| **mean_dist** | Distance moyenne des éclairs à l’aéroport | Proximité du foyer par rapport à la zone sensible. |
| **std_dist** | Dispersion des distances | Orage étalé vs concentré. |
| **mean_maxis** | Moyenne de `maxis` (erreur de localisation, km) | Donne une échelle spatiale / incertitude ; utile pour caractériser la « taille » du phénomène. |
| **max_maxis** | Max de `maxis` sur l’alerte | Capte le pire cas d’étalement / incertitude. |
| **storm_size_km** | Grand axe approximatif de l’orage (km), à partir de la boîte lon/lat des éclairs | **Variable créée** : un orage « long » (ligne de grains, gros amas) dure souvent plus longtemps. |

### 3.3 Variables dérivées créées

| Variable | Définition | Rôle |
|----------|------------|------|
| **density** | `n_lightnings / (mean_maxis + 0.01)` | **Créée** : intensité relative — beaucoup d’éclairs sur une zone à faible maxis = orage intense et concentré ; peu d’éclairs sur forte maxis = orage diffus. |
| **last_lightning_is_cg** | 1 si le **dernier** éclair de l’alerte (par date) est nuage–sol, 0 sinon | **Créée** : si le dernier éclair est encore au sol, l’orage peut être encore actif (intérêt pour la phase de fin d’orage). |

### 3.4 Variables temporelles (contexte)

| Variable | Rôle |
|----------|------|
| **start_year, start_month, start_dayofyear, start_hour** | Saisonnalité et heure de début ; les orages n’ont pas la même dynamique selon le moment. |

### 3.5 Variables catégorielles

| Variable | Rôle |
|----------|------|
| **airport** | Aéroport (one-hot) — chaque lieu a un climat et des types d’orages différents. |
| **cluster** | Type d’orage (0 à 3) issu du K-Means, si on utilise `alerts_preprocessed_with_cluster.csv` — résume un profil (bref / moyen / long / très actif) et améliore légèrement la MAE. |

---

## 4. Clustering (types d’orages)

- **Algorithme :** K-Means (k = 4), après standardisation des variables.
- **Données :** `alerts_preprocessed.csv` (une ligne = une alerte).
- **Variables utilisées pour le clustering :** durée, nombre d’éclairs (total, CG, IC), amplitude moyenne, distance (moyenne, std), maxis (moyenne, max), density, storm_size_km.
- **Objectif :** obtenir 4 profils interprétables, par ex. « petit orage bref », « orage moyen actif », « orage long », « système très long et très actif ». Ces profils servent à l’**analyse et à l’explication** (oral, rapport). Le label **cluster** peut en plus être utilisé comme variable catégorielle du modèle de régression pour gagner un peu en MAE.
- **Sortie :** affichage en terminal (effectifs, centres, résumé par cluster) + fichier `alerts_preprocessed_with_cluster.csv` (alertes + colonne `cluster`).

---

## 5. Modèle de régression (prédiction de la durée)

- **Cible :** `duration_total_minutes`.
- **Algorithme :** Random Forest (par défaut, meilleur MAE) ; on compare aussi Extra Trees et Gradient Boosting (éventuellement XGBoost si installé).
- **Pipeline :** (1) StandardScaler sur les variables numériques, (2) OneHotEncoder sur les catégorielles (airport, et cluster si présent), (3) Random Forest (ou autre) en régression.
- **Évaluation :** validation croisée 5-fold ; métrique principale = MAE (min).
- **Fonctionnement :** le modèle apprend à prédire la durée totale à partir des variables listées ci-dessus ; aucune variable ne dépend de la fin réelle de l’alerte (pas de fuite). Les prédictions sont ensuite utilisées pour calculer des probabilités de fin d’orage (voir ci-dessous).

---

## 6. Sorties du modèle (probabilités et décision)

- **Fichier de prédictions :** `advanced_model_predictions.csv` (durée prédite, durée réelle si disponible).
- **Sigma (incertitude) :** écart-type des résidus (prédit − réel) pour interpréter la prédiction comme une loi normale.
- **Probabilité par minute :** pour chaque minute t (ex. 0 à 120), on calcule P(fin d’orage avant t min) via la fonction de répartition normale (moyenne = prédiction, écart-type = sigma). Affichage dans le terminal (`probability_per_minute.py`).
- **Aide à la décision :** pour un seuil de confiance (ex. 95 %), on indique à partir de quelle minute on peut lever l’alerte (médiane sur les alertes). Comparaison « sans modèle » (règle 30 min) vs « avec modèle (95 %) » : gain en minutes (médiane et moyenne).
- **Graphique :** `plot_probability_per_minute.py` génère `graph_decision_minutes_vs_percentage.png` (courbe confiance % en fonction des minutes + tableau comparatif).

---

## 7. Résultats principaux

- **MAE (CV 5-fold) :** ~11,46 min (avec type d’orage / cluster) ; ~11,56 min sans cluster.
- **RMSE :** ~24,4 min.
- **R² :** ~0,76.
- **Part des prédictions à ±10 min du réel :** ~68 %.

---

## 8. Lancer le projet

```bash
pip install -r requirements.txt
python3 preprocessing_databattle_2026.py
python3 clustering_storm_types.py
python3 advanced_modeling_databattle_2026.py
python3 probability_per_minute.py
python3 plot_probability_per_minute.py
```

Le CSV brut doit être présent dans `data_train_databattle2026/segment_alerts_all_airports_train.csv`. Si `alerts_preprocessed_with_cluster.csv` n’existe pas, le modèle s’entraîne sans la variable `cluster`.

---

## 9. Fichiers du dépôt (résumé)

| Fichier | Rôle |
|---------|------|
| `preprocessing_databattle_2026.py` | Brut → tableau par alerte + variables créées (density, storm_size_km, last_lightning_is_cg, etc.). |
| `clustering_storm_types.py` | K-Means sur les alertes → 4 types d’orages + export CSV avec `cluster`. |
| `advanced_modeling_databattle_2026.py` | Entraînement régression (RF/ET/GBR), comparaison, sauvegarde des prédictions. |
| `probability_per_minute.py` | Probabilités par minute et comparaison avant/après modèle (terminal). |
| `plot_probability_per_minute.py` | Graphique de décision (minutes vs confiance %). |
| `journal_projet_databattle_2026.md` | Journal détaillé du projet. |
| `presentation_sujet_lundi_2_mars.pdf` | Sujet Data Battle. |
| `info_data.md` | Description des champs du jeu de données brut. |

Les CSV et PNG générés (prétraitement, prédictions, graphiques) sont listés dans `.gitignore` ; les scripts permettent de les régénérer.
