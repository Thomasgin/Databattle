## Journal de projet – Data Battle 2026

### 1. Compréhension du sujet et des données

- 2026-03-09  – Lecture de la présentation Meteorage (`presentation_sujet_lundi_2_mars.pdf`) et de la page officielle de l’évènement ([Data Battle 2026 - Association IA Pau](https://iapau.org/events/data-battle-2026/)).
- 2026-03-09  – Objectif identifié : modéliser **probabilistiquement** la fin d’une alerte d’orage (moment où l’on peut lever l’alerte) à partir de l’historique des éclairs dans un rayon de 20 km autour d’un aéroport.
- 2026-03-09  – Lecture des consignes techniques (fichier `Info data.docx` converti en `info_data.md`) décrivant précisément le dataset :
  - Une ligne = **un éclair** avec ses caractéristiques spatio-temporelles.
  - Colonnes principales : `date`, `lon`, `lat`, `amplitude`, `maxis`, `icloud`, `dist`, `azimuth`, `lightning_id`, `lightning_airport_id`, `alert_airport_id`, `is_last_lightning_cloud_ground`.
  - Les champs `alert_airport_id` et `is_last_lightning_cloud_ground` ne sont renseignés que pour les éclairs à moins de 20 km de l’aéroport.

### 2. Plan de travail initial

1. Exploration des données :
   - Vérifier les types, valeurs manquantes, distributions de base.
   - Examiner la structure des alertes (`alert_airport_id`) et l’identification du dernier éclair nuage-sol (`is_last_lightning_cloud_ground`).
2. Définition du problème d’apprentissage :
   - Formuler la prédiction de fin d’alerte comme un problème **temps jusqu’à l’événement** (survie / temps de fin d’orage) ou comme une **probabilité de fin dans un horizon donné**.
   - Choisir une granularité temporelle (par exemple segmenter en pas de temps).
3. Prétraitement :
   - Nettoyage des données (filtrage éventuel des valeurs aberrantes, gestion des données de Pise 2016 pour les éclairs intra-nuage si nécessaire).
   - Construction de caractéristiques (features) à partir de l’historique des éclairs : intensité, rythme d’occurrence, temps depuis le dernier éclair, features spatiales (distance/azimut), séparation nuage-sol / intra-nuage, etc.
4. Modélisation :
   - Construire un ou plusieurs modèles probabilistes (par ex. modèles de survie, modèles de risque instantané ou modèles probabilistes paramétriques / non paramétriques).
   - Validation croisée par aéroport, évaluation par des métriques adaptées (log-loss, Brier score, métriques de survie, etc. selon le règlement).
5. Analyse par aéroport :
   - Étudier les différences structurelles d’orages selon les aéroports (fréquence, durée, types d’orage).
6. Restitution :
   - Préparer un notebook/script reproductible.
   - Documenter les choix et résultats pour la présentation finale.

### 3. Étapes à venir (backlog)

- Créer un script Python pour l’exploration (`exploration_databattle_2026.py`).
- Mettre en place une première pipeline de prétraitement.
- Définir précisément la cible (variable à prédire) selon le règlement complet et les colonnes disponibles.
- Prototyper au moins un modèle baseline simple (ex : modèle de type survie ou régression logistique sur features temporelles).

### 4. Décisions techniques

- 2026-03-09 – Tous les développements seront réalisés en **Python** sous forme de scripts (`.py`) afin de faciliter l’exécution en ligne de commande et la reproductibilité.
- 2026-03-09 – Un premier script d’exploration sera créé : `exploration_databattle_2026.py`, chargé de :
  - Lire le fichier `data_train_databattle2026/segment_alerts_all_airports_train.csv`.
  - Inspecter les types de colonnes et la présence de valeurs manquantes.
  - Calculer quelques statistiques descriptives globales et par aéroport.
  - Sauvegarder ces résumés dans des fichiers `.csv` ou `.txt` pour pouvoir les commenter dans le rapport.

### 5. Prétraitement – plan

- 2026-03-09 – Décision de créer un script `preprocessing_databattle_2026.py` chargé de :
  - Lire à nouveau le fichier brut `segment_alerts_all_airports_train.csv`.
  - Clarifier la colonne d’identifiant d’alerte (`airport_alert_id`, qui correspond à `alert_airport_id` dans la documentation).
  - Construire un tableau agrégé par alerte contenant :
    - l’aéroport,
    - l’identifiant d’alerte,
    - la date de début d’alerte (premier éclair de l’alerte),
    - la date du dernier éclair nuage-sol (si marquée),
    - la durée de l’alerte (en minutes),
    - le nombre total d’éclairs, le nombre d’éclairs nuage-sol et intra-nuage.
  - Sauvegarder ce tableau dans un fichier `alerts_preprocessed.csv` qui servira de base pour l’analyse et la modélisation.

### 6. Exploration des données (résultats initiaux)

- 2026-03-09 – Exécution de `exploration_databattle_2026.py` :
  - Le fichier contient **507 071 éclairs** au total.
  - Toutes les colonnes principales sont complètes (aucune valeur manquante) sauf :
    - `airport_alert_id` : 56 599 valeurs non nulles (les lignes à moins de 20 km de l’aéroport).
    - `is_last_lightning_cloud_ground` : 56 599 valeurs non nulles (même logique que ci-dessus).
  - Le timestamp `date` a été converti en type datetime (UTC).
  - Un fichier `summary_global.csv` a été généré contenant les statistiques descriptives (moyenne, min, max, etc.) pour toutes les colonnes.
  - Un fichier `summary_airport_counts.csv` a été généré avec le nombre d’éclairs par aéroport :
    - Pise : 156 718 éclairs.
    - Bastia : 125 919 éclairs.
    - Biarritz : 115 191 éclairs.
    - Ajaccio : 72 501 éclairs.
    - Nantes : 36 742 éclairs.
  - Les colonnes d’alerte utilisées par le script (`airport_alert_id`, `is_last_lightning_cloud_ground`) semblent légèrement différentes de celles décrites dans `info_data.md` (`alert_airport_id` vs `airport_alert_id`), ce qui sera clarifié lors de la prochaine étape de traitement (alignement de la nomenclature des colonnes).

### 7. Prétraitement – résultats

- 2026-03-09 – Exécution de `preprocessing_databattle_2026.py` :
  - Le script recharge les données brutes et renomme la colonne `airport_alert_id` en `alert_airport_id` pour coller à la documentation.
  - Il ne conserve que les lignes avec un identifiant d’alerte (`alert_airport_id` non nul), soit **56 599 éclairs** associés à des alertes dans le rayon de 20 km.
  - Il construit un tableau agrégé `alerts_preprocessed.csv` au niveau **alerte** (une ligne par alerte) contenant :
    - `airport`, `alert_airport_id`.
    - `n_lightnings` : nombre total d’éclairs dans l’alerte.
    - `start_time`, `end_time` : premier et dernier éclair observés pour cette alerte.
    - `n_cloud_ground`, `n_intra_cloud` : nombre d’éclairs nuage-sol et intra-nuage.
    - `last_cloud_ground_time` : date du dernier éclair (actuellement égale à `end_time` car tous les éclairs marqués sont dans la zone d’alerte).
    - `duration_total_minutes`, `duration_until_last_cg_minutes` : durées en minutes de l’alerte globale et jusqu’au dernier éclair nuage-sol.
    - Quelques features calendaires (`start_year`, `start_month`, `start_dayofyear`, `start_hour`).
  - Résumé : le tableau agrégé contient **2 627 alertes distinctes** réparties sur **5 aéroports**, avec une durée médiane d’environ **8,9 minutes**, une durée maximale d’environ **579 minutes**.

### 8. Définition de la cible et approche de modélisation

- 2026-03-09 – Choix initial de la **cible** pour le modèle :
  - On utilise la variable `duration_total_minutes` issue de `alerts_preprocessed.csv`, qui représente la durée totale d’une alerte (du premier au dernier éclair détecté dans la zone).
  - Objectif de ce premier modèle : prédire cette durée totale à partir d’informations disponibles au début de l’alerte (ou très tôt dans l’alerte), pour approcher la « fin d’orage ».
- 2026-03-09 – Approche de modélisation de base :
  - Construire un modèle de **régression** (par exemple Gradient Boosting ou Random Forest) sur des features agrégées par alerte (`n_lightnings`, `n_cloud_ground`, `n_intra_cloud`, `start_year`, `start_month`, `start_dayofyear`, `start_hour`, encodage de l’aéroport).
  - Évaluer la qualité de prédiction en termes de MAE / RMSE sur un jeu de validation.
  - Interpréter le modèle comme un premier modèle **probabiliste simple** en supposant une distribution (par exemple gaussienne) autour de la prédiction moyenne, ce qui permettra ensuite d’en déduire une probabilité que l’alerte soit terminée avant un certain horizon de temps.
- 2026-03-09 – Décision de créer un script `modeling_databattle_2026.py` qui :
  - Charge `alerts_preprocessed.csv`.
  - Sépare les données en entraînement / validation (en prenant en compte au minimum un mélange par aéroport).
  - Entraîne un modèle de régression et calcule des métriques d’erreur.
  - Sauvegarde les prédictions et résumés de performance pour analyse ultérieure.

### 9. Modélisation – premier modèle baseline

- 2026-03-09 – Création du script `modeling_databattle_2026.py` :
  - Utilisation de `scikit-learn` avec un `Pipeline` combinant :
    - un `ColumnTransformer` pour :
      - standardiser les variables numériques (`n_lightnings`, `n_cloud_ground`, `n_intra_cloud`, `start_year`, `start_month`, `start_dayofyear`, `start_hour`),
      - encoder en one-hot la variable catégorielle `airport`.
    - un `RandomForestRegressor` (300 arbres, `random_state=42`, `n_jobs=-1`) comme modèle de régression.
  - La cible utilisée est `duration_total_minutes`.
  - Séparation des données en train/validation via `train_test_split` (80 % / 20 %, mélange aléatoire).
- 2026-03-09 – Résultats sur le jeu de validation :
  - MAE ≈ 13,45 minutes.
  - RMSE ≈ 27,24 minutes.
  - Écart-type des résidus (sigma) ≈ 27,12 minutes.
  - Les prédictions de validation (features + `duration_true` + `duration_pred` + `residual`) sont enregistrées dans `model_validation_predictions.csv`.
- Interprétation probabiliste simple :
  - À ce stade, on peut approximer la durée prédite d’une alerte comme une variable aléatoire de type \( \mathcal{N}(\hat{y}, \sigma^2) \) où :
    - \( \hat{y} \) est la prédiction moyenne du modèle,
    - \( \sigma \) est l’écart-type global des résidus estimé sur le jeu de validation (~27 minutes),
  - Ce qui permet d’estimer la probabilité que l’alerte soit terminée avant un certain horizon \( t \) en utilisant la fonction de répartition de la loi normale (approche à raffiner avec un modèle de survie plus avancé si nécessaire).

### 10. Probabilités de fin d’alerte – plan

- 2026-03-09 – Décision de calculer explicitement, pour chaque alerte du jeu de validation, la **probabilité que l’alerte soit terminée avant différents horizons temporels** :
  - Horizons choisis (en minutes) : 10, 20, 30, 45, 60.
  - Utilisation de l’approximation gaussienne \( \mathcal{N}(\hat{y}, \sigma^2) \) avec :
    - \( \hat{y} \) : `duration_pred` issue du modèle Random Forest,
    - \( \sigma \) : écart-type global des résidus estimé sur la validation (~27,12 minutes).
  - Pour chaque horizon \( t \), calcul de \( P(\text{fin} \le t) = \Phi\left(\frac{t - \hat{y}}{\sigma}\right) \) où \( \Phi \) est la CDF de la loi normale standard.
- 2026-03-09 – Décision de créer un script `probabilities_databattle_2026.py` qui :
  - Charge `model_validation_predictions.csv`.
  - Recalcule \( \sigma \) à partir de la colonne `residual`.
  - Ajoute une colonne de probabilité pour chaque horizon (ex. `p_end_before_10`, `p_end_before_20`, etc.).
  - Sauvegarde le résultat dans `validation_probabilities.csv`.

### 11. Analyse par aéroport – plan

- 2026-03-09 – Pour répondre au deuxième axe du sujet (analyse des tendances d’orages par aéroport) :
  - Construire des statistiques par aéroport (nombre d’alertes, durées moyenne/médiane, distribution des types d’éclair, saisonnalité simple par mois).
  - Identifier d’éventuelles différences structurelles (par ex. orages plus longs ou plus fréquents sur certains aéroports).
- Décision de créer un script `analysis_airports_databattle_2026.py` qui :
  - Charge `alerts_preprocessed.csv`.
  - Calcule un tableau de synthèse par aéroport (`analysis_airport_summary.csv`).
  - Calcule un tableau par aéroport et par mois (`analysis_airport_monthly.csv`) pour visualiser les tendances saisonnières.

### 12. Probabilités de fin d’alerte – résultats

- 2026-03-09 – Création et exécution de `probabilities_databattle_2026.py` :
  - Le script charge `model_validation_predictions.csv`, recalcule \(\sigma\) à partir de `residual` (~27,12 minutes).
  - Pour chaque alerte de la validation, il crée les colonnes :
    - `p_end_before_10`, `p_end_before_20`, `p_end_before_30`, `p_end_before_45`, `p_end_before_60`.
  - Les probabilités sont stockées dans `validation_probabilities.csv`.
  - Exemple de comportement :
    - Pour une alerte de durée prédite ≈ 19,5 minutes, la probabilité que l’alerte soit terminée avant 30 minutes est d’environ 0,65, et avant 60 minutes d’environ 0,93.
    - Pour une alerte de durée prédite ≈ 75 minutes, la probabilité que l’alerte soit terminée avant 30 minutes est faible (~0,05) mais augmente pour des horizons plus longs (≈0,29 pour 60 minutes).

### 13. Analyse par aéroport – résultats

- 2026-03-09 – Exécution de `analysis_airports_databattle_2026.py` :
  - Le script produit deux fichiers :
    - `analysis_airport_summary.csv` : synthèse globale par aéroport.
    - `analysis_airport_monthly.csv` : synthèse par aéroport et par mois (nombre d’alertes et durée moyenne).
  - Résumé par aéroport (valeurs moyennes arrondies) :
    - Ajaccio : ~530 alertes, durée moyenne ≈ 28,5 min, médiane ≈ 8,4 min, max ≈ 400 min, ~20 éclairs par alerte.
    - Bastia : ~532 alertes, durée moyenne ≈ 32,2 min, médiane ≈ 10,1 min, max ≈ 498 min, ~26 éclairs par alerte.
    - Biarritz : ~590 alertes, durée moyenne ≈ 25,6 min, médiane ≈ 7,1 min, max ≈ 388 min, ~17 éclairs par alerte.
    - Nantes : ~206 alertes, durée moyenne ≈ 27,9 min, médiane ≈ 8,8 min, max ≈ 382 min, ~21 éclairs par alerte.
    - Pise : ~769 alertes, durée moyenne ≈ 33,1 min, médiane ≈ 11,3 min, max ≈ 579 min, ~23 éclairs par alerte.
  - Ces résultats suggèrent que Pise et Bastia présentent en moyenne des alertes plus longues, tandis que Biarritz a des alertes généralement plus courtes, ce qui alimente l’analyse des spécificités locales demandée dans le sujet.

### 14. Modèles avancés – plan d’optimisation

- 2026-03-09 – Objectif : améliorer la précision du modèle de durée d’alerte en testant plusieurs algorithmes de régression d’arbres et un réglage d’hyperparamètres :
  - Comparer plusieurs modèles :
    - Random Forest (baseline actuelle).
    - Extra Trees (forêt extrêmement aléatoire).
    - Gradient Boosting.
  - Utiliser une **validation croisée en K-fold (K=5)** pour obtenir des métriques plus stables.
  - Effectuer un **RandomizedSearchCV** sur une Random Forest pour trouver un jeu d’hyperparamètres mieux adapté.
- 2026-03-09 – Décision de créer un script `advanced_modeling_databattle_2026.py` qui :
  - Charge `alerts_preprocessed.csv`.
  - Construit le même jeu de features que précédemment.
  - Définit plusieurs pipelines `scikit-learn` (préprocesseur + modèle).
  - Calcule, pour chaque modèle, la MAE et la RMSE moyennes en validation croisée.
  - Entraîne le meilleur modèle sur tout l’échantillon d’entraînement et sauvegarde un tableau de comparaison (`advanced_model_comparison.csv`) ainsi que les prédictions finales de ce meilleur modèle (`advanced_model_predictions.csv`).

### 15. Modèles avancés – résultats

- 2026-03-09 – Création et exécution de `advanced_modeling_databattle_2026.py` :
  - Trois modèles de base ont été évalués par validation croisée (K=5) via `cross_val_predict` :
    - `rf_default` (Random Forest, 300 arbres),
    - `et_default` (Extra Trees, 400 arbres),
    - `gbr_default` (Gradient Boosting, 500 arbres, learning_rate = 0,05, profondeur max = 3).
  - Un **RandomizedSearchCV** a ensuite été lancé sur une Random Forest (`rf_tuned`) avec 20 combinaisons d’hyperparamètres, `cv=5`, critère de score = `neg_mean_absolute_error`.
- 2026-03-09 – Tableau de comparaison (`advanced_model_comparison.csv`) :
  - `rf_tuned` : MAE CV ≈ **12,71 min**, RMSE CV ≈ **26,16 min**.
  - `rf_default` : MAE CV ≈ 12,96 min, RMSE CV ≈ 26,69 min.
  - `gbr_default` : MAE CV ≈ 13,44 min, RMSE CV ≈ 27,40 min.
  - `et_default` : MAE CV ≈ 13,59 min, RMSE CV ≈ 27,87 min.
- 2026-03-09 – Meilleur modèle retenu :
  - Le meilleur modèle est la Random Forest réglée (`rf_tuned`) avec hyperparamètres :
    - `n_estimators` = 300,
    - `max_depth` = 20,
    - `min_samples_split` = 2,
    - `min_samples_leaf` = 1,
    - `max_features` = `sqrt`.
  - Ce modèle offre une amélioration nette par rapport au baseline initial (MAE ≈ 13,45 min → 12,71 min ; RMSE ≈ 27,24 min → 26,16 min).
  - Les prédictions complètes du meilleur modèle sur l’ensemble des alertes sont sauvegardées dans `advanced_model_predictions.csv` (colonnes : features + `duration_true` + `duration_pred_best`).
- **Ajout de XGBoost** : le script `advanced_modeling_databattle_2026.py` inclut désormais un modèle **XGBoost** (`xgb_default`) dans la comparaison (500 arbres, learning_rate 0,05, max_depth 6). Si la librairie `xgboost` est installée (`pip install xgboost` ou `pip install -r requirements.txt`), il est évalué en validation croisée avec les autres modèles ; le meilleur (au sens de la MAE) est alors retenu pour les prédictions finales. Un fichier `requirements.txt` liste les dépendances du projet (pandas, numpy, scikit-learn, xgboost).

### 16. Mise en dépôt Git et organisation du projet

- 2026-03-09 – Initialisation du dépôt Git local dans le dossier du projet (`/home/cytech/Desktop/DataBattle`) :
  - `git init`
  - Ajout d’un fichier `.gitignore` pour ne pas versionner le gros fichier de données brutes fourni par l’organisateur :
    - `data_train_databattle2026/segment_alerts_all_airports_train.csv`
  - Cela permet de garder un dépôt léger tout en ayant la structure des données (dossier + fichiers de description) dans le versionnement.
- 2026-03-09 – Connexion au dépôt GitHub existant `https://github.com/Thomasgin/Databattle.git` :
  - Ajout du remote : `git remote add origin https://github.com/Thomasgin/Databattle.git`
  - Récupération des branches distantes : `git fetch origin`
  - Bascule sur la branche de travail : `git checkout -b hugo origin/hugo`
- 2026-03-09 – Ajout de tous les fichiers du projet et création d’un commit :
  - Fichiers ajoutés (principaux) :
    - Scripts Python : `exploration_databattle_2026.py`, `preprocessing_databattle_2026.py`, `modeling_databattle_2026.py`, `advanced_modeling_databattle_2026.py`, `probabilities_databattle_2026.py`, `analysis_airports_databattle_2026.py`.
    - Fichiers de résultats : `alerts_preprocessed.csv`, `summary_global.csv`, `summary_airport_counts.csv`, `model_validation_predictions.csv`, `validation_probabilities.csv`, `advanced_model_comparison.csv`, `advanced_model_predictions.csv`, `analysis_airport_summary.csv`, `analysis_airport_monthly.csv`.
    - Fichiers de description : `journal_projet_databattle_2026.md`, `info_data.md`, `Info data.docx`, `presentation_sujet_lundi_2_mars.pdf`.
    - Fichier `.gitignore`.
  - Commit créé avec le message : **« Ajout du projet complet Data Battle 2026 »**.
- 2026-03-09 – Poussée sur GitHub :
  - La commande `git push origin hugo` est prête à être exécutée (l’échec observé localement est uniquement dû à l’absence de saisie des identifiants GitHub sur la machine).
  - Une fois le push effectué avec les identifiants GitHub de l’utilisateur, la branche `hugo` du dépôt `Databattle` contiendra l’intégralité du projet tel que décrit dans ce journal.

### 17. Récapitulatif global des étapes (de A à Z)

1. **Compréhension du sujet et des données**
   - Lecture des supports Meteorage (PDF de présentation et page officielle de l’évènement) et du document de description des données (`Info data.docx` → `info_data.md`).
   - Reformulation de l’objectif : prédire la fin d’alerte d’orage de manière probabiliste à partir de données d’éclairs autour de six aéroports européens.
2. **Planification**
   - Définition d’un plan de travail : exploration → prétraitement → choix de la cible → modèles → probabilités de fin → analyse par aéroport → mise en dépôt.
3. **Exploration des données brutes (`exploration_databattle_2026.py`)**
   - Chargement du CSV brut `segment_alerts_all_airports_train.csv`.
   - Conversion de `date` en `datetime`.
   - Vérification des types et des valeurs manquantes.
   - Sauvegarde de statistiques globales (`summary_global.csv`) et du nombre d’éclairs par aéroport (`summary_airport_counts.csv`).
4. **Prétraitement au niveau alerte (`preprocessing_databattle_2026.py`)**
   - Renommage de `airport_alert_id` en `alert_airport_id` pour correspondre à la documentation.
   - Sélection des éclairs appartenant à une alerte (`alert_airport_id` non nul).
   - Agrégation au niveau alerte : premières / dernières dates d’éclair, nombre d’éclairs, séparation nuage-sol / intra-nuage, durées en minutes, features calendaires.
   - Sauvegarde du tableau d’alertes dans `alerts_preprocessed.csv`.
5. **Définition de la cible et premier modèle (`modeling_databattle_2026.py`)**
   - Cible : `duration_total_minutes` (durée totale de l’alerte).
   - Features : compteurs d’éclairs, dates (année, mois, jour de l’année, heure) et aéroport encodé.
   - Modèle baseline : Random Forest + préprocessing (scaling + one-hot).
   - Split train/validation, calcul des métriques (MAE, RMSE, sigma des résidus) et sauvegarde des prédictions de validation (`model_validation_predictions.csv`).
6. **Construction de probabilités de fin d’alerte (`probabilities_databattle_2026.py`)**
   - Approximation de la durée comme loi normale \(\mathcal{N}(\hat{y}, \sigma^2)\).
   - Calcul, pour chaque alerte de validation, de la probabilité que l’alerte soit terminée avant différents horizons (10, 20, 30, 45, 60 min).
   - Sauvegarde des résultats dans `validation_probabilities.csv`.
7. **Analyse des tendances par aéroport (`analysis_airports_databattle_2026.py`)**
   - Synthèse globale par aéroport : nombre d’alertes, durées moyenne/médiane/max, nombre moyen d’éclairs.
   - Analyse par aéroport et par mois : nombre d’alertes et durée moyenne pour étudier les tendances saisonnières.
   - Sauvegarde dans `analysis_airport_summary.csv` et `analysis_airport_monthly.csv`.
8. **Optimisation avancée des modèles (`advanced_modeling_databattle_2026.py`)**
   - Comparaison en validation croisée (K=5) de plusieurs modèles d’arbres : Random Forest, Extra Trees, Gradient Boosting, et XGBoost (si installé).
   - RandomizedSearchCV sur la Random Forest pour affiner les hyperparamètres.
   - Sélection du meilleur modèle (`rf_tuned`) sur la base de la MAE et de la RMSE.
   - Sauvegarde du tableau de comparaison (`advanced_model_comparison.csv`) et des prédictions complètes du meilleur modèle (`advanced_model_predictions.csv`).
9. **Mise sous contrôle de version (Git + GitHub)**
   - Initialisation du dépôt Git local, création du `.gitignore`, connexion au dépôt GitHub `Thomasgin/Databattle` sur la branche `hugo`.
   - Commit unique résumant l’ajout du projet complet.
   - Préparation du push vers GitHub (à finaliser avec les identifiants de l’utilisateur).

Ce récapitulatif permet de suivre pas à pas l’évolution du projet, depuis la compréhension du sujet jusqu’au modèle final optimisé et à la mise en dépôt, tout en reliant chaque étape aux scripts et fichiers produits.

### 18. Améliorations pour gagner des minutes (sans overlifting)

- 2026-03-09 – Objectif : améliorer les prédictions pour réduire le temps d'attente tout en maîtrisant l'**overlifting** (lever l'alerte avant la fin réelle de l'orage).
- **Nouvelles features** (sans fuite de données) : dans `preprocessing_databattle_2026.py`, agrégation par alerte : `mean_dist`, `std_dist`, `mean_amplitude`. Pas de `lightning_rate` (éviter fuite : la durée est la cible).
- **Comparaison baseline vs amélioré** (`compare_baseline_improved.py`) : baseline = features d'origine ; amélioré = baseline + mean_dist, std_dist, mean_amplitude. Même RF, 5-fold CV. Métriques : MAE, RMSE, sigma, gain médian vs règle 30 min, **taux d'overlifting** (cible ~5 % pour seuil 95 %).
- **Résultats** (`comparison_baseline_improved.csv`) : Baseline MAE ≈ 12,96 min, overlifting ≈ 3,9 %. Amélioré MAE ≈ 12,30 min, overlifting ≈ 3,9 %. À sécurité égale, le modèle amélioré réduit MAE et sigma. Le modèle final utilise les features améliorées.
- **Meilleure version pour temps d'alerte minimal** : dans `advanced_modeling_databattle_2026.py`, comparaison de rf_tuned, et_tuned, gbr_tuned (plus modèles de base). Sélection automatique du modèle à **plus faible MAE** (temps d'alerte le plus court à complexité raisonnable). Actuellement retenu : **rf_tuned** (MAE CV ≈ 12,22 min). Une seule pipeline (préprocesseur + un modèle), pas d’ensemble lourd.
