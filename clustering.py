import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
def run_clustering(path_csv):
    df = pd.read_csv(path_csv)
    
    features = [
        'ic_ratio',        # Physique de l'éclair
        'mean_amplitude',  # Puissance du système
        'speed',           # Cinématique (mouvement)
        'lat_std',         # Étalement Nord-Sud
        'lon_std',         # Étalement Est-Ouest
        'mean_dist',       # Centrage par rapport à l'aéroport
        'prox_ratio'       # Dangerosité immédiate
    ]
    
    # Suppression des lignes avec des NaN sur ces colonnes si besoin
    df_clean = df.dropna(subset=features).copy()
    
    # préparation du clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[features])
    
    # On fixe à 3 clusters pour une typologie claire
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clean['storm_type'] = kmeans.fit_predict(X_scaled)
    
    # visualisation avec PCA 
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(X_scaled)
    df_clean['pca_1'] = pca_results[:, 0]
    df_clean['pca_2'] = pca_results[:, 1]
    
    # affichage des clusters
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Graphique A : Projection PCA 
    sns.scatterplot(
        data=df_clean, x='pca_1', y='pca_2', hue='storm_type', 
        palette='bright', alpha=0.6, ax=ax1, s=60, edgecolor='black'
    )
    ax1.set_title('Topologie des Orages (Projection PCA)', fontsize=15, fontweight='bold')
    ax1.set_xlabel('Dimension 1 : Diversité Structurelle')
    ax1.set_ylabel('Dimension 2 : Intensité/Vitesse')
    
    # Graphique B : Signature par Cluster (Z-Scores)
    cluster_profiles = pd.DataFrame(X_scaled, columns=features)
    cluster_profiles['storm_type'] = df_clean['storm_type'].values
    z_score_means = cluster_profiles.groupby('storm_type').mean()
    
    sns.heatmap(z_score_means, annot=True, cmap='RdYlBu_r', center=0, ax=ax2)
    ax2.set_title('Signature Physique par Cluster (Z-Score)', fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('cluster_analysis.png')
    plt.show()
    
    return df_clean

if __name__ == "__main__":
    #df_clean = load_and_clean('alerts_preprocessed.csv')
    #df_clustered = run_clustering(df_clean)
    #df_clustered.to_csv('alerts_with_clusters.csv', index=False)
    #print("Fichier 'alerts_with_clusters.csv' généré avec succès.")
    df_visu = run_clustering('alerts_with_clusters.csv')