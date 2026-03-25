from sklearn.decomposition import PCA

def acp(df):
    features = ['prox_ratio', 'ic_ratio', 'speed', 'log_storm_surface', 
                'mean_amplitude', 'log_n_lightnings', 'mean_dist']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    # On garde 3 pour l'instant
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['storm_type'] = kmeans.fit_predict(X_scaled)
    
    # --- AJOUT PCA POUR VISUALISATION ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]
    
    plt.figure(figsize=(15, 5))
    
    # Nouveau Graphique 1 : Vue PCA (La réalité multidimensionnelle)
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=df, x='pca_1', y='pca_2', hue='storm_type', palette='viridis')
    plt.title('Séparation réelle (Vue PCA)')
    
    # Ton graphique actuel (Vue métier)
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df, x='log_n_lightnings', y='log_storm_surface', 
                    hue='storm_type', palette='viridis', alpha=0.6)
    plt.title('Intensité vs Taille')
    
    # Heatmap
    plt.subplot(1, 3, 3)
    cluster_means = df.groupby('storm_type')[features].mean()
    sns.heatmap(cluster_means, annot=True, cmap='YlGnBu')
    plt.title('Signature Physique')
    
    plt.tight_layout()
    plt.show()
