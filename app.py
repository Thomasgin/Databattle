import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Météorage - Alerte Orage", page_icon="⚡", layout="wide")

# --- FONCTIONS DE CHARGEMENT DES DONNÉES ---
# C'est ici que tu fais le lien avec le travail de tes camarades !
@st.cache_data # Permet de ne pas recharger les données à chaque clic
def load_data():
    # lire le fichier généré  : df = pd.read_csv("Databattle/advanced_model_predictions.csv")
    
    # Pour l'instant, on crée de fausses données pour tester l'interface
    data = pd.DataFrame({
        'airport': ['Bron', 'Bastia', 'Nantes', 'Ajaccio', 'Pise', 'Biarritz'],
        'lat': [45.7294, 42.5527, 47.1532, 41.9236, 43.695, 43.4683],
        'lon': [4.9389, 9.4837, -1.6107, 8.8029, 10.399, -1.524],
        'probabilite_fin': [96, 45, 12, 88, 99, 5], # La probabilité calculée par le modèle
        'temps_ecoule_min': [25, 12, 2, 18, 32, 1]
    })
    return data

# Charger les données
df = load_data()

# --- BARRE LATÉRALE (SIDEBAR) --- l'image n'existe pas
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Logo_m%C3%A9t%C3%A9orage.jpg/960px-Logo_m%C3%A9t%C3%A9orage.jpg", width=200)
st.sidebar.title("Paramètres")
aeroport_choisi = st.sidebar.selectbox("Sélectionner un aéroport", df['airport'].unique())

# Filtrer les données pour l'aéroport choisi
infos_aeroport = df[df['airport'] == aeroport_choisi].iloc[0]

# --- EN-TÊTE DU DASHBOARD ---
st.title(f"⚡ Suivi Orage - Aéroport de {aeroport_choisi}")
st.markdown("Interface d'aide à la décision pour la levée d'alerte météorologique.")

# --- SECTION 1 : LES KPI (Indicateurs clés) ---
# On définit un code couleur selon la probabilité
proba = infos_aeroport['probabilite_fin']
if proba > 95:
    statut, couleur, message = "🟢 VERT", "green", "Alerte levée. Reprise des opérations."
elif proba > 75:
    statut, couleur, message = "🟠 ORANGE", "orange", "Orage en dissipation. Préparation des équipes."
else:
    statut, couleur, message = "🔴 ROUGE", "red", "Orage actif. Opérations suspendues."

# Affichage des 3 colonnes d'indicateurs
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Statut actuel", value=statut, delta=message, delta_color=couleur)
with col2:
    st.metric(label="Probabilité de fin d'orage", value=f"{proba} %")
with col3:
    st.metric(label="Temps depuis le dernier impact", value=f"{infos_aeroport['temps_ecoule_min']} min")


st.divider()

# --- SECTION 2 : CARTE & GRAPHIQUE ---
col_map, col_graph = st.columns([1, 1]) # Deux colonnes de largeur égale

with col_map:
    st.subheader("📍 Carte des derniers impacts")
    # Création de la carte centrée sur l'aéroport
    m = folium.Map(location=[infos_aeroport['lat'], infos_aeroport['lon']], zoom_start=11)
    # Marqueur de l'aéroport
    folium.Marker(
        [infos_aeroport['lat'], infos_aeroport['lon']], 
        popup=aeroport_choisi, 
        icon=folium.Icon(color='blue', icon='plane')
    ).add_to(m)
    
    # Dessiner la zone de surveillance (ex: 20km)
    folium.Circle(
        location=[infos_aeroport['lat'], infos_aeroport['lon']],
        radius=20000, # 20 km
        color='red',
        fill=True,
        fill_opacity=0.1
    ).add_to(m)
    
    # Afficher la carte dans Streamlit
    st_folium(m, width=500, height=400)

with col_graph:
    st.subheader("📈 Évolution de la probabilité")
    # Ici utiliser la sortie du fichier "probabilite_par_minute.py" de ton équipe !
    # Simulation de données pour le graphique :
    temps = list(range(0, infos_aeroport['temps_ecoule_min'] + 5))
    # Création d'une courbe logarithmique fictive pour simuler le modèle
    probas_historiques = [min(99, i * (proba / max(1, infos_aeroport['temps_ecoule_min']))) for i in temps] 
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(temps, probas_historiques, marker='o', color='blue')
    ax.axhline(y=95, color='r', linestyle='--', label='Seuil de sécurité (95%)')
    ax.set_xlabel("Minutes depuis le dernier éclair")
    ax.set_ylabel("Probabilité de fin (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    st.pyplot(fig)

# --- SECTION 3 : GAIN ÉCONOMIQUE ---
st.success(f"💡 **Bilan IA :** Avec la règle standard de 30 minutes, il faudrait encore attendre {max(0, 30 - infos_aeroport['temps_ecoule_min'])} minutes. Notre modèle propose une décision sécurisée **maintenant**.")
