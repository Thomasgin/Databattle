# Soutenance - Modele prediction duree orage (4 min, 2 personnes)

Objectif : tenir une presentation **100% sur le modele de prediction de duree**, en **4 minutes** (2 min chacun), claire et sans chevauchement.

---

## 1) Timing exact

- **Intro commune : 15 s**
- **Personne 1 : 1 min 45**
- **Transition : 5 s**
- **Personne 2 : 1 min 45**
- **Conclusion commune : 10 s**

Total : **4 min**

---

## 2) Repartition - uniquement la partie modele

## Personne 1 (2 min) - Construction et evaluation du modele

### Script conseille (chronometre)

1. **Probleme (20 s)**
   - "On predit la duree d'alerte orage en minutes."
   - "La cible est `duration_total_minutes`, sinon `duration_minutes`."

2. **Modeles compares (45 s)**
   - **`linear_baseline`** : regression lineaire avec `StandardScaler` (pipeline `scaler + LinearRegression`).
   - **`rf_default`** : `RandomForestRegressor` avec `n_estimators=120` (sans tuning).
   - **`rf_tuned`** : `RandomForestRegressor` + `RandomizedSearchCV` (tuning sur hyperparametres, `n_iter=6`, CV interne sur les memes groupes).
   - **`xgb_tuned`** : `XGBRegressor` + `RandomizedSearchCV` (uniquement si `xgboost` est installe ; `n_iter=6`).

   *(Le script utilise aussi `get_dummies` : toutes les colonnes sauf la cible deviennent des features tabulaires (encodage des categoriels).)*

3. **Evaluation robuste (35 s)**
   - Erreur mesuree sur des predictions **out-of-fold** (pas de fit sur tout le dataset puis predict dessus).
   - Si `alert_airport_id` existe : `GroupKFold` (3 plis) pour ne pas melanger des lignes d'une meme alerte entre train/val.
   - Si tuning (`rf_tuned` / `xgb_tuned`) : tuning fait **a l'interieur des plis train** (nested CV via re-entraînement par pli).

4. **Phrase de cloture P1 (5 s)**
   - "On compare ces 4 modeles avec une evaluation sans fuite."

### Message cle P1
"On privilegie une evaluation fiable, pas seulement un bon score apparent."

---

## Personne 2 (2 min) - Resultats, sortie du modele, limites

### Script conseille (chronometre)

1. **Resultats (45 s)**
   - Le script calcule **MAE** et **RMSE** a partir des predictions out-of-fold.
   - Les candidats peuvent etre : `linear_baseline`, `rf_default`, `rf_tuned`, `xgb_tuned` (si disponible).
   - **Modele choisi** : le script retient le candidat avec la **MAE la plus faible** en CV (`best_name = min(results, key=MAE)`).
   - Phrase : "On privilégie la MAE car c'est l'erreur moyenne absolue sur la duree (en minutes)."
   - (A dire a l'oral) "Dans notre run, `best_name` vaut : ____ (affiche par le terminal)."

2. **Sortie concrete du modele (25 s)**
   - Fichier produit : `advanced_model_predictions.csv`.
   - Colonnes : `duration_true` et `duration_pred_best`.
   - Important : `duration_pred_best` correspond aux **predictions out-of-fold du meilleur modele** (pas des predictions apprises sur le meme dataset).

3. **Interet projet (20 s)**
   - Prediction exploitable pour estimer la duree reelle d'une alerte.
   - Base pour une aide a la decision ulterieure.

4. **Limites + suite (15 s)**
   - Limite : clustering fait globalement en amont (risque transductif residuel).
   - Suite : clustering par pli train-only et enrichissement des variables.

5. **Phrase de cloture P2 (5 s)**
   - "Donc on obtient une estimation de duree plus fiable, choisie sur MAE CV."

### Message cle P2
"Le modele ne donne pas seulement un chiffre : il fournit une estimation exploitable, evaluee proprement."

---

## 3) Slides minimales (4 slides, 1 min par slide)

1. **Objectif + cible du modele**
2. **Modeles testes + protocole d'evaluation**
3. **Resultats MAE/RMSE + meilleur modele**
4. **Sortie fichier + limites + ameliorations**

---

## 4) Intro, transition, conclusion (phrases pretes)

### Intro commune (15 s)
"On presente la partie modele qui predit la duree des alertes orage, avec une comparaison de plusieurs algorithmes et une evaluation sans fuite."

### Transition P1 -> P2 (5 s)
"Je laisse [Prenom 2] presenter les resultats et ce qu'on en tire."

### Conclusion commune (10 s)
"Notre contribution est un modele de duree robuste, mesurable et directement exploitable."

---

## 5) Checklist repetition (special 4 min)

- [ ] Chacun respecte strictement 2 min.
- [ ] Vous dites clairement MAE, RMSE, OOF et GroupKFold.
- [ ] Vous citez `advanced_model_predictions.csv`.
- [ ] Vous avez 1 limite + 1 amelioration en une phrase.
- [ ] Une repetition complete en 4 min chrono.

---

## 6) Points detailles a ne pas oublier a l'oral

Ces points prennent 20-30 secondes chacun si le jury demande plus de detail.

1. **Pourquoi plusieurs modeles et pas un seul ?**
   - Pour avoir une baseline simple (lineaire) et des modeles non lineaires (RF/XGB).
   - Le but est de comparer des families differentes sous le meme protocole.

2. **Pourquoi MAE comme critere principal ?**
   - Unite interpretable directement en minutes.
   - Plus robuste aux valeurs extremes que RMSE.
   - RMSE est garde pour completer l'analyse des grosses erreurs.

3. **Pourquoi GroupKFold est important ici ?**
   - Plusieurs lignes peuvent appartenir a la meme alerte (`alert_airport_id`).
   - Sans grouping, des lignes proches d'une meme alerte pourraient etre a la fois en train et val (fuite d'information).

4. **Pourquoi des predictions out-of-fold ?**
   - Chaque ligne est predite par un modele qui ne l'a jamais vue.
   - Evite l'optimisme artificiel d'un `fit(X)` puis `predict(X)`.

5. **Ce que contient exactement la sortie du modele**
   - `advanced_model_predictions.csv` :
     - `duration_true` = duree reelle
     - `duration_pred_best` = prediction OOF du meilleur modele (selon MAE)

---

## 7) Questions probables du jury + reponses courtes

### Q1. "Quel modele final avez-vous retenu ?"
**Reponse :**
"Le script choisit automatiquement le modele avec la MAE CV la plus faible. Sur notre execution, le modele retenu est `____`."

### Q2. "Pourquoi ne pas garder uniquement XGBoost ?"
**Reponse :**
"On ne suppose pas a l'avance qu'un modele est meilleur. On compare objectivement baseline, RF et XGB avec la meme methode d'evaluation, puis on selectionne au score."

### Q3. "Comment evitez-vous la fuite de donnees ?"
**Reponse :**
"On utilise des predictions out-of-fold, GroupKFold par alerte quand possible, et le tuning est fait a l'interieur des plis train."

### Q4. "Pourquoi 3 plis et pas 5 ou 10 ?"
**Reponse :**
"C'est un compromis precision/temps de calcul, surtout avec tuning nested CV. 3 plis donne deja une estimation stable pour comparer les modeles."

### Q5. "A quoi sert la regression lineaire si elle est souvent moins bonne ?"
**Reponse :**
"Elle sert de reference interpretable. Si un modele complexe ne bat pas la baseline, sa complexite n'est pas justifiee."

### Q6. "Quelle est votre principale limite sur cette partie modele ?"
**Reponse :**
"Le clustering est calcule globalement en amont, ce qui peut introduire une fuite transductive residuelle via `storm_type`."

### Q7. "Que feriez-vous en priorite pour ameliorer ?"
**Reponse :**
"Premier axe : recalculer le clustering par pli d'entrainement uniquement. Deuxieme axe : enrichir les variables temporelles."

### Q8. "Comment savoir si le modele est exploitable metier ?"
**Reponse :**
"On regarde la MAE en minutes (interpretable) et on verifie que les predictions OOF restent stables; ensuite on peut fixer des seuils d'acceptabilite metier."

---

## 8) Reponses longues (si jury insiste)

- **Sur le choix MAE vs RMSE**
  - "MAE mesure l'erreur moyenne absolue en minutes, donc tres lisible pour l'operationnel.
    RMSE penalise davantage les grosses erreurs. On suit les deux, mais la decision finale est basee sur la MAE."

- **Sur la robustesse de la selection**
  - "Le meilleur modele est choisi sur predictions OOF, pas sur apprentissage complet.
    Cela reduit le risque de sur-optimisme et donne une performance plus proche du reel."

- **Sur la reproductibilite**
  - "Le code fixe `random_state=42` et applique le meme schema CV a tous les modeles, pour des comparaisons coherentes."

---

## 9) Empreinte carbone (a dire + questions jury)

## Message court a dire (20-30 s)

"On a integre une logique de sobriete de calcul : on compare un nombre limite de modeles, avec un tuning borne (`n_iter=6`) et une validation a 3 plis. Le calcul est volontairement sequentiel (`n_jobs=1`) pour eviter la surconsommation CPU/RAM. On cherche donc le meilleur compromis entre performance predictive et cout de calcul."

## Ce que vous pouvez montrer comme choix concrets dans le code

- CV limitee a 3 plis (`cv_splits = 3`) au lieu de 5/10.
- Tuning limite (`RandomizedSearchCV`, `n_iter=6`) au lieu de grilles tres larges.
- Pas de deep learning lourd pour cette version (modeles tabulaires plus sobres).
- Execution sequentielle (`OUTER_N_JOBS = 1`) pour limiter surcharge machine.

## Questions jury frequentes sur l'impact environnemental

### Q1. "Pourquoi parler d'empreinte carbone ici ?"
**Reponse :**
"Parce que l'entrainement de modeles a un cout energetique. Meme sur un projet etudiant, on peut appliquer des choix de sobriete algorithmique."

### Q2. "Qu'avez-vous fait concretement pour la reduire ?"
**Reponse :**
"On a borne le nombre d'essais d'hyperparametres, limite le nombre de plis, et evite les modeles tres lourds tant qu'ils ne sont pas necessaires."

### Q3. "Pourquoi ne pas faire plus de tuning si cela peut ameliorer le score ?"
**Reponse :**
"Au-dela d'un certain point, le gain de MAE devient marginal alors que le cout de calcul augmente fortement. On cherche un compromis performance/energie."

### Q4. "Comment ameliorer encore cet aspect ?"
**Reponse :**
"On peut faire de l'early stopping (sur XGBoost), reduire l'espace de recherche selon les premiers resultats, et n'entrainer les modeles lourds que si la baseline est insuffisante."

## Formule utile (si jury insiste)

"Notre approche est 'frugale by design' : baseline d'abord, complexite ensuite seulement si le gain est significatif."

---

## 10) Ce qu'il ne faut pas oublier en soutenance (check final)

- [ ] Dire explicitement **quel modele est retenu** sur votre run (`best_name`).
- [ ] Donner au moins **une valeur MAE** et **une valeur RMSE** de votre execution.
- [ ] Expliquer **pourquoi MAE est le critere de selection**.
- [ ] Expliquer **OOF + GroupKFold** en une phrase simple.
- [ ] Citer la **limite principale** (clustering global en amont).
- [ ] Ajouter la phrase **empreinte carbone** (sobriete de calcul).

