# Environnement et social - Data Battle IA PAU 2026

## 1) Objectif de cette note
Documenter les impacts environnementaux et sociaux du projet, et les actions concrètes mises en place pour améliorer le compromis performance / impact.

## 2) Mesure et estimation des impacts

### 2.1 Indicateurs déjà mesurés dans le code
- `compute_footprint_proxy.csv` : temps de calcul par modèle (proxy de consommation).
- `model_benchmark_report.csv` : performance + temps (permet de comparer efficacité vs coût de calcul).
- Exécution volontairement séquentielle (`n_jobs=1`) pour limiter les pics CPU/RAM.

### 2.2 Interprétation
- Les modèles les plus complexes (XGBoost/CatBoost/MLP) coûtent plus de temps de calcul.
- Les gains de performance doivent être mis en balance avec ce coût.
- Le choix final doit donc intégrer une logique de sobriété (pas seulement la meilleure MAE brute).

## 3) Cycle de vie (analyse qualitative)

### 3.1 Matériel et infrastructure
- Exécution locale (pas d'infra cloud lourde obligatoire).
- Réduction de l'empreinte réseau et de la dépendance à des services externes.

### 3.2 Logiciels et maintenance
- Stack Python standard (portable, maintenable, faible verrouillage techno).
- Scripts modulaires (`clustering.py`, `modele.py`, `probabilite_par_minute.py`).

### 3.3 Exploitation
- Possibilité d'exécuter uniquement les modèles nécessaires (évite des runs inutiles).
- Possibilité de désactiver des modèles lourds selon le contexte d'usage.

## 4) Effets rebond et risques indirects

- **Risque 1 : sur-automatisation** de la décision de levée d'alerte.
  - Mitigation : conserver un contrôle humain + seuils de confiance explicites.
- **Risque 2 : multiplication des runs** pour gagner marginalement en score.
  - Mitigation : protocole de benchmark stable et budget de calcul défini.
- **Risque 3 : faux sentiment de sécurité** sur cas rares.
  - Mitigation : suivi de métriques de sécurité dédiées (cas critiques proches aéroport).

## 5) Actions déjà mises en place (green coding)

- Architecture modulaire et simple.
- Validation croisée robuste mais calibrée (3 plis).
- Tuning borné (espace et itérations limités selon modèle).
- Export de rapports pour éviter de relancer inutilement les expériences.

## 6) Plan d'amélioration à court terme

1. Ajouter une mesure énergétique outillée (plateforme Gaia / CodeCarbon) sur un run de référence.
2. Définir un budget calcul (temps max/run) et une règle de sélection "MAE + coût calcul".
3. Conserver une configuration "léger en production" (modèles rapides) et "complet en R&D".
4. Mettre en place un journal d'expériences pour éviter les exécutions redondantes.

## 7) Dimension sociale et parties prenantes

### 7.1 Parties prenantes
- Opérateurs aéroportuaires / décisionnaires sécurité.
- Étudiants et enseignants (transfert de compétences).
- Utilisateurs finaux impactés par les décisions d'alerte.

### 7.2 Apports
- Aide à la décision plus explicite et traçable.
- Meilleure compréhension des facteurs qui influencent les durées d'alerte.

### 7.3 Précautions
- Ne pas remplacer l'expertise métier : le modèle assiste, il ne décide pas seul.
- Documenter les limites et les cas où la confiance du modèle est faible.
