import pandas as pd

print("Chargement des fichiers...")
# On prend le fichier RÉPARÉ par le script intracloud
df_fixed = pd.read_csv('alerts_preprocessed_fixed.csv')
df_raw = pd.read_csv('segment_alerts_all_airports_train.csv')

# Détection automatique du nom de la colonne d'alerte dans le brut
raw_col = 'airport_alert_id' if 'airport_alert_id' in df_raw.columns else 'alert_airport_id'

print(f"Calcul des colonnes spatiales (mean_dist, std_dist, mean_amplitude)...")

# 1. Calculer les statistiques spatiales à partir du brut
spatial_features = df_raw.groupby(['airport', raw_col]).agg(
    mean_dist=('dist', 'mean'),
    std_dist=('dist', 'std'),
    mean_amplitude=('amplitude', 'mean')
).reset_index()

# 2. Renommer pour la fusion
spatial_features = spatial_features.rename(columns={raw_col: 'alert_airport_id'})

# 3. Fusionner avec ton fichier réparé
print("Fusion finale...")
df_final = pd.merge(df_fixed, spatial_features, on=['airport', 'alert_airport_id'], how='left')

# 4. Nettoyage des std_dist (0 si un seul éclair)
df_final['std_dist'] = df_final['std_dist'].fillna(0)

# 5. SAUVEGARDE FINALE sous le nom attendu par le modèle
df_final.to_csv('alerts_preprocessed.csv', index=False)
print("TERMINÉ ! Le fichier 'alerts_preprocessed.csv' est maintenant RÉPARÉ et COMPLET.")