import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse

# --- 0. CONFIGURATION DE LA CHARTE GRAPHIQUE ---
sns.set_theme(style="white")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'figure.facecolor': 'white'
})
# Palette : Bleu (Solution), Rouge (Brut/Danger), Gris (Contexte)
PALETTE = {"blue": "#1f77b4", "red": "#d62728", "grey": "#7f7f7f"}

# Chargement des données
df_raw = pd.read_csv('segment_alerts_all_airports_train.csv')
df_prep = pd.read_csv('alerts_preprocessed.csv')
df_raw['date'] = pd.to_datetime(df_raw['date'])

# =================================================================
# GRAPHE 1 : CARTE DE SITUATION (Ancrage Métier)
# =================================================================
def plot_map():
    plt.figure(figsize=(10, 7))
    coords = df_raw.groupby('airport')[['lat', 'lon']].mean()
    
    plt.scatter(coords['lon'], coords['lat'], s=250, c=PALETTE['red'], marker='^', label='Aéroports')
    for i, txt in enumerate(coords.index):
        plt.annotate(txt, (coords.lon.iloc[i]+0.2, coords.lat.iloc[i]), fontsize=11, fontweight='bold')
    
    plt.title("Périmètre d'Étude : 6 Aéroports Européens", pad=20)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig('1_carte_situation.png', dpi=300, bbox_inches='tight')
    print("Graphe 1 généré.")

# =================================================================
# GRAPHE 2 : NAISSANCE DE L'ALERTE (Preprocessing Technique)
# =================================================================
def plot_alert_birth():
    # Sélection d'un exemple à Nantes
    sample = df_prep[df_prep['airport'] == 'Nantes'].iloc[5]
    raw_sample = df_raw[(df_raw['airport'] == 'Nantes') & 
                        (df_raw['date'] >= pd.to_datetime(sample['start_time'])) & 
                        (df_raw['date'] <= pd.to_datetime(sample['end_time']))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    
    # Gauche : Brut
    ax1.scatter(raw_sample['lon'], raw_sample['lat'], c=PALETTE['red'], s=40, alpha=0.7)
    ax1.set_title("1. Données Brutes\n(Impacts Isolés)", fontweight='bold')
    
    # Droite : Groupé
    ax2.scatter(raw_sample['lon'], raw_sample['lat'], c='lightgrey', s=40, alpha=0.3)
    ellipse = Ellipse((sample['lon_mean'], sample['lat_mean']), 
                      width=sample['lon_std']*4, height=sample['lat_std']*4,
                      angle=0, facecolor=PALETTE['blue'], alpha=0.3, label='Alerte Unique')
    ax2.add_patch(ellipse)
    ax2.scatter(sample['lon_mean'], sample['lat_mean'], color=PALETTE['blue'], s=100, marker='*', label='Centre')
    ax2.set_title("2. Données Groupées\n(L'Épisode Orageux)", fontweight='bold', color=PALETTE['blue'])
    
    plt.suptitle("Transformation Technique : Du Point à l'Événement", fontsize=18)
    plt.savefig('2_naissance_alerte.png', dpi=300, bbox_inches='tight')
    print("Graphe 2 généré.")

# =================================================================
# GRAPHE 3 : MATRICE DE NETTOYAGE (Feature Engineering)
# =================================================================
def plot_matrices():
    # Sélection des colonnes numériques
    cols_raw = ['lon', 'lat', 'amplitude', 'maxis', 'dist', 'azimuth']
    cols_prep = ['duration_minutes', 'n_lightnings', 'lightning_per_minute', 'mean_amplitude', 'ic_ratio', 'mean_dist']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # MATRICE 1 (Brut) : Mêmes couleurs (coolwarm) et même échelle (-1 à 1)
    sns.heatmap(df_raw[cols_raw].corr(), annot=True, cmap='coolwarm', fmt=".2f", 
                ax=ax1, vmin=-1, vmax=1, center=0, cbar=False)
    ax1.set_title("1. Structure des Données Brutes\n(Peu d'informations corrélées)", pad=15)
    
    # MATRICE 2 (Prep) : Mêmes couleurs (coolwarm) et même échelle (-1 à 1)
    sns.heatmap(df_prep[cols_prep].corr(), annot=True, cmap='coolwarm', fmt=".2f", 
                ax=ax2, vmin=-1, vmax=1, center=0, cbar=False)
    ax2.set_title("2. Structure après Feature Engineering\n(Corrélations métier révélées)", pad=15, color='#1f77b4', fontweight='bold')
    
    plt.suptitle("Comparaison de la Richesse Informationnelle", fontsize=20, y=1.05)
    plt.savefig('3_matrice_nettoyage.png', dpi=300, bbox_inches='tight')
    print("Graphe 3 généré (avec échelle de couleurs identique).")
# Exécution
plot_map()
plot_alert_birth()
plot_matrices()