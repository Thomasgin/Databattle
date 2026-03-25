import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def evaluate_storm_clustering(csv_path):
    # 1. Chargement des données
    df = pd.read_csv(csv_path)
    
    # 2. Préparation des features (Même logique que ton code précédent)
    df['speed'] = np.sqrt(df['lat_delta']**2 + df['lon_delta']**2)
    df['storm_surface'] = df['lat_std'] * df['lon_std']
    df['log_storm_surface'] = np.log1p(df['storm_surface'])
    df['log_n_lightnings'] = np.log1p(df['n_lightnings'])
    
    features = ['prox_ratio', 'ic_ratio', 'speed', 'log_storm_surface', 
                'mean_amplitude', 'log_n_lightnings', 'mean_dist']
    
    X = df[features]
    
    # 3. Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Application du K-Means (K=3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # --- 5. CALCUL DES MÉTRIQUES ---
    print("--- ÉVALUATION DU CLUSTERING (K=3) ---")
    
    # Score de Silhouette (Qualité de la séparation)
    # Proche de 1 = Parfait, Proche de 0 = Chevauchement
    sil_score = silhouette_score(X_scaled, labels)
    
    # Score de Calinski-Harabasz (Plus il est élevé, mieux c'est)
    ch_score = calinski_harabasz_score(X_scaled, labels)
    
    print(f"Score de Silhouette      : {sil_score:.3f}")
    print(f"Score Calinski-Harabasz  : {ch_score:.1f}")
    print("-" * 38)
    
    # Interprétation rapide
    if sil_score > 0.5:
        print("Résultat : Excellente séparation des types d'orages.")
    elif sil_score > 0.3:
        print("Résultat : Structure raisonnable, quelques zones de flou.")
    else:
        print("Résultat : Clusters très proches. Essayez d'ajouter des variables.")

if __name__ == "__main__":
    evaluate_storm_clustering('alerts_preprocessed.csv')