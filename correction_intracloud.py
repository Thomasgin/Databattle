import pandas as pd

print("Chargement des fichiers...")
df_alerts = pd.read_csv('alerts_preprocessed.csv')
df_raw = pd.read_csv('segment_alerts_all_airports_train.csv')

# Conversion des dates pour pouvoir comparer
df_alerts['start_time'] = pd.to_datetime(df_alerts['start_time'])
df_alerts['end_time'] = pd.to_datetime(df_alerts['end_time'])
df_raw['date'] = pd.to_datetime(df_raw['date'])

# 2. Filtrer le brut pour ne garder que les intra-nuages (True)
df_ic = df_raw[df_raw['icloud'] == True].copy()

print("Réparation de la colonne n_intra_cloud...")

def count_ic(row):
    # On cherche les éclairs IC qui sont au même aéroport ET entre le début et la fin de l'alerte
    mask = (df_ic['airport'] == row['airport']) & \
           (df_ic['date'] >= row['start_time']) & \
           (df_ic['date'] <= row['end_time'])
    return mask.sum()

# Appliquer la réparation
df_alerts['n_intra_cloud'] = df_alerts.apply(count_ic, axis=1)

# 3. Mettre à jour le nombre total d'éclairs (n_lightnings = CG + IC)
df_alerts['n_lightnings'] = df_alerts['n_cloud_ground'] + df_alerts['n_intra_cloud']

# 4. Sauvegarder le fichier réparé
df_alerts.to_csv('alerts_preprocessed_fixed.csv', index=False)
print("Terminé ! Ton fichier 'alerts_preprocessed_fixed.csv' est maintenant correct.")