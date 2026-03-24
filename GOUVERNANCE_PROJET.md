# Gouvernance projet et poursuite - Data Battle IA PAU 2026

## 1) Processus de décision

### 1.1 Règles de décision technique
- Une évolution est acceptée si elle améliore le compromis:
  - performance (MAE/RMSE),
  - robustesse (OOF, GroupKFold, absence de fuite),
  - coût de calcul (temps d'entraînement).

### 1.2 Sources pour trancher
- `model_benchmark_report.csv` pour comparer les modèles.
- `compute_footprint_proxy.csv` pour estimer le coût calcul.
- `compute_footprint_estimate_simple.csv` pour une estimation indicative énergie / CO₂eq (sans outil externe).
- `model_explainability_top_features.csv` pour l'interprétabilité.

## 2) Rôles (adaptables selon disponibilité)
- **Référent données** : qualité des données, cohérence des variables.
- **Référent modèle** : entraînement, validation, benchmark.
- **Référent impact** : environnement/social, documentation des risques.
- **Référent produit** : besoins métier, restitution et démonstration.

## 3) Gestion de projet

### 3.1 Rythme recommandé
- 1 point hebdomadaire court (décisions, blocages, priorités).
- 1 run de référence par sprint pour comparer les versions.

### 3.2 Critères de "done"
- Résultats reproductibles.
- Rapport benchmark à jour.
- Explicabilité disponible pour le meilleur modèle.
- Documentation mise à jour (README + impacts + décisions).

## 4) Plan de poursuite après le Data Battle

### 4.1 Court terme (1-2 semaines)
1. Figer une configuration de référence (modèles activés, seed, seuil de décision).
2. Archiver les totaux de `compute_footprint_estimate_simple.csv` ; compléter avec Gaia ou CodeCarbon si exigé.
3. Ajouter un tableau de bord simple de suivi des runs.

### 4.2 Moyen terme (1-2 mois)
1. Renforcer l'évaluation sur cas critiques proches aéroport (ex: <= 3 km).
2. Ajouter une validation temporelle stricte pour simuler la production.
3. Formaliser la politique "agir tôt vs attendre" via seuils de confiance métier.

### 4.3 Mise en routine
- Utilisation en mode standard par des étudiants/professionnels:
  - commande simple de lancement,
  - livrables CSV exploitables,
  - guide d'interprétation court.

## 5) Risques projet et mitigation
- **Risque de dérive des données** : monitoring périodique + ré-entraînement planifié.
- **Risque de complexité excessive** : préférer modèles plus légers si gain marginal.
- **Risque d'usage hors contexte** : documentation des limites et garde-fous décisionnels.
