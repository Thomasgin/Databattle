import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 1. CHARGEMENT ET NETTOYAGE ---
def load_and_clean(path):
    df = pd.read_csv(path)
    df_c = df.copy()
    
    # Feature Engineering
    df_c['speed'] = np.sqrt(df_c['lat_delta']**2 + df_c['lon_delta']**2)
    df_c['storm_surface'] = df_c['lat_std'] * df_c['lon_std']
    
    # Encodage cyclique
    df_c['hour_sin'] = np.sin(2 * np.pi * df_c['start_hour'] / 24)
    df_c['hour_cos'] = np.cos(2 * np.pi * df_c['start_hour'] / 24)
    
    # Log-transform pour stabiliser la variance
    cols_to_log = ['n_lightnings', 'lightning_per_minute', 'storm_surface']
    for col in cols_to_log:
        df_c[f'log_{col}'] = np.log1p(df_c[col])

    # Clipping
    for col in ['mean_amplitude', 'lightning_per_minute']:
        limit = df_c[col].quantile(0.99)
        df_c[col] = np.clip(df_c[col], None, limit)
        
    return df_c

# --- 2. CLUSTERING ET VISUALISATION ---
def run_clustering(df):
    features = ['prox_ratio', 'ic_ratio', 'speed', 'log_storm_surface', 
                'mean_amplitude', 'log_n_lightnings', 'mean_dist']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['storm_type'] = kmeans.fit_predict(X_scaled)
    
    # --- VISUALISATION ---
    plt.figure(figsize=(12, 5))
    
    # Graphique 1 : Intensité vs Taille
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='log_n_lightnings', y='log_storm_surface', 
                    hue='storm_type', palette='viridis', alpha=0.6)
    plt.title('Typologie des Orages (Intensité vs Taille)')
    
    # Graphique 2 : Profil des Clusters (Moyennes)
    plt.subplot(1, 2, 2)
    cluster_means = df.groupby('storm_type')[features].mean()
    sns.heatmap(cluster_means, annot=True, cmap='YlGnBu')
    plt.title('Signature Physique par Cluster')
    
    plt.tight_layout()
    plt.savefig('cluster_analysis.png')
    plt.show()
    
    return df

if __name__ == "__main__":
    df_clean = load_and_clean('alerts_preprocessed.csv')
    df_clustered = run_clustering(df_clean)
    df_clustered.to_csv('alerts_with_clusters.csv', index=False)
    print("Fichier 'alerts_with_clusters.csv' généré avec succès.")