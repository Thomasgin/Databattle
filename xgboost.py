import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Chargement des données
df = pd.read_csv('alerts_with_clusters.csv')

# 2. Séparation des variables (X) et de la cible (y)
# On garde les 13 variables pour l'instant (incluant le cluster)
X = df.drop(['lightning_id', 'lightning_airport_id','airport_alert_id'], axis=1, errors='ignore')
y = df['is_last_lightning_cloud_ground']

# 3. Prétraitement
# Conversion des variables textuelles en nombres (One-Hot Encoding)
X = pd.get_dummies(X)

# Si votre cible 'y' est du texte (ex: "A", "B"), on la transforme en 0, 1...
if y.dtype == 'boolean':
    le = LabelEncoder()
    y = le.fit_transform(y)

# 4. Division en ensembles d'entraînement et de test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Création et entraînement du modèle XGBoost
# On utilise des paramètres robustes par défaut
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# On prédit les probabilités :
y_pred = model.predict_proba(X_test)

print("--- RÉSULTATS DU MODÈLE ---")
print(f"Précision globale : {accuracy_score(y_test, y_pred):.2%}")
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# 7. ANALYSE : Quelle variable est la plus importante ?
# C'est ici que vous verrez si votre variable 'cluster' est utile
print("\nAffichage de l'importance des variables...")
plt.figure(figsize=(10, 8))
xgb.plot_importance(model, importance_type='weight', title='Importance des 13 variables')
plt.show()