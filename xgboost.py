import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error

# --- 1. PRÉPARATION ---
df = pd.read_csv('alerts_with_clusters.csv')
df_final = pd.get_dummies(df, columns=['airport'], prefix='apt')

features_clustering = ['prox_ratio', 'ic_ratio', 'speed', 'log_storm_surface', 
                        'mean_amplitude', 'log_n_lightnings', 'mean_dist']
features_xgb = features_clustering + ['hour_sin', 'hour_cos', 'storm_type']
airport_cols = [col for col in df_final.columns if col.startswith('apt_')]

X = df_final[features_xgb + airport_cols]
y_duration = df_final['duration_minutes']
y_risk = (df_final['duration_minutes'] > 20).astype(int)

X_train, X_test, y_train_risk, y_test_risk = train_test_split(X, y_risk, test_size=0.2, random_state=42)
_, _, y_train_time, y_test_time = train_test_split(X, y_duration, test_size=0.2, random_state=42)

# --- 2. ENTRAÎNEMENT ---
# Modèle de Classification (Probabilité de persistance)
model_risk = xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
model_risk.fit(X_train, y_train_risk)

# Modèle de Régression (Temps restant)
model_time = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
model_time.fit(X_train, y_train_time)

# --- 3. ÉVALUATION ET PRÉDICTION ---
def get_storm_prediction(index):
    data_point = X.iloc[[index]]
    prob_risk = model_risk.predict_proba(data_point)[0][1]
    est_minutes = model_time.predict(data_point)[0]
    cluster_id = df.iloc[index]['storm_type']
    
    print(f"\n--- RÉSULTAT IA (ORAGE TYPE {int(cluster_id)}) ---")
    print(f"Durée estimée : {est_minutes:.1f} minutes")
    print(f"Confiance de l'alerte (Risque >20min) : {prob_risk*100:.1f}%")
    
    verdict = "MAINTIEN ALERTE" if prob_risk > 0.15 else "LEVÉE ALERTE"
    print(f"VERDICT : {verdict}")

if __name__ == "__main__":
    # Petit check de performance sur le set de test
    preds_risk = model_risk.predict(X_test)
    print("Performance Modèle Risque :")
    print(classification_report(y_test_risk, preds_risk))
    
    # Test sur un exemple
    get_storm_prediction(0)