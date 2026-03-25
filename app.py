import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import math

# --- FONCTIONS MATHÉMATIQUES DU MODÈLE ---
# Issue du fichier probabilite_par_minute.py de l'équipe
def normal_cdf(x):
    """Calcule la probabilité cumulée (loi normale)"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Météorage - Alerte Orage", page_icon="⚡", layout="wide")

# --- CHARGEMENT DES VRAIES DONNÉES ---
# Remplace ton ancienne fonction load_data() par celle-ci :

@st.cache_data
def load_data():
    try:
        # 1. On charge le fichier des prédictions (les 2 colonnes)
        df_preds = pd.read_csv("advanced_model_predictions.csv")
        
        # 2. On charge le fichier source de tes camarades pour "voler" la colonne airport
        # Remplace le nom du fichier par celui que vous utilisez (ex: alerts_preprocessed.csv ou alerts_final_model.csv)
        fichier_source = "alerts_preprocessed.csv" 
        
        try:
            df_source = pd.read_csv(fichier_source)
            # On vérifie que la colonne existe bien dans la source et qu'elle a la même taille
            if 'airport' in df_source.columns and len(df_source) == len(df_preds):
                df_preds['airport'] = df_source['airport'] # On fusionne magiquement !
            else:
                st.warning(f"⚠️ La colonne 'airport' est introuvable dans {fichier_source} ou les tailles ne correspondent pas.")
        except FileNotFoundError:
            st.warning(f"⚠️ Fichier source '{fichier_source}' introuvable pour récupérer les aéroports. Utilisation de fausses données d'aéroport pour la démo.")
            # Solution de secours absolue pour ne pas que l'IHM crashe devant le jury
            # On distribue les aéroports au hasard juste pour avoir une interface qui tourne
            import random
            faux_aeroports = ['Bron', 'Bastia', 'Nantes', 'Ajaccio', 'Pise', 'Biarritz']
            df_preds['airport'] = [random.choice(faux_aeroports) for _ in range(len(df_preds))]
            
        return df_preds
        
    except FileNotFoundError:
        st.error("⚠️ Fichier 'advanced_model_predictions.csv' introuvable.")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # Dictionnaire des coordonnées des aéroports (car le CSV final n'a peut-être pas gardé lat/lon exacts)
    coords_aeroports = {
        'Bron': [45.7294, 4.9389],
        'Bastia': [42.5527, 9.4837],
        'Ajaccio': [41.9236, 8.8029],
        'Nantes': [47.1532, -1.6107],
        'Pise': [43.695, 10.399],
        'Biarritz': [43.4683, -1.524]
    }

    # --- BARRE LATÉRALE (SIDEBAR) ---
   # st.sidebar.markdown("# 🌩️ **MÉTÉORAGE**")
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Logo_m%C3%A9t%C3%A9orage.jpg/960px-Logo_m%C3%A9t%C3%A9orage.jpg", width=200)
    st.sidebar.markdown("---")
    
    # Sélection de l'aéroport (on vérifie s'il y a bien une colonne airport)
    liste_aeroports = df['airport'].unique() if 'airport' in df.columns else list(coords_aeroports.keys())
    aeroport_choisi = st.sidebar.selectbox("📍 Sélectionner un aéroport", liste_aeroports)
    
    print("liste aéroport : \n", liste_aeroports)
    print("aéroport choisi : \n", aeroport_choisi)

    # Filtrer les données pour l'aéroport choisi
    if 'airport' in df.columns:
        df_aeroport = df[df['airport'] == aeroport_choisi].reset_index(drop=True)
    else:
        df_aeroport = df.copy() # Sécurité si la colonne a sauté
        
    print("df_aeroport : \n", df_aeroport)

    # --- CALCUL DES PERFORMANCES SPÉCIFIQUES À L'AÉROPORT ---
    # On calcule l'erreur MAE et le Sigma (écart-type) just pour cet aéroport !
    residus = df_aeroport['duration_true'] - df_aeroport['duration_pred_best']
    mae_aeroport = np.mean(np.abs(residus))
    sigma_aeroport = np.std(residus)
    # Sécurité au cas où le modèle serait "parfait" (sigma = 0)
    if sigma_aeroport == 0: sigma_aeroport = 1.0
    
    print("residus : \n", residus)
    print("mae_aeroport : \n", mae_aeroport)
    print("sigma_aeroport : \n", sigma_aeroport)

    # --- SÉLECTION D'UNE ALERTE POUR LA SIMULATION ---
    st.sidebar.markdown("### ⏱️ Simulateur de temps")
    # On prend la première alerte de cet aéroport comme exemple à simuler
    alerte_simulee = df_aeroport.iloc[0]
    prediction_modele = alerte_simulee['duration_pred_best']
    duree_reelle = alerte_simulee['duration_true']
    
    print("alerte_simulee : \n", alerte_simulee)
    print("prediction_modele : \n", prediction_modele)
    print("duree_reelle : \n", duree_reelle)

    # Le curseur magique pour la soutenance !
    temps_ecoule = st.sidebar.slider(
        "Minutes écoulées depuis le dernier éclair :", 
        min_value=0, 
        max_value=120, 
        value=int(prediction_modele), # Par défaut, on se place sur la prédiction
        step=1
    )

    # Calcul de la probabilité actuelle selon le temps écoulé sur le slider
    valeur_z = (temps_ecoule - prediction_modele) / sigma_aeroport
    proba_actuelle = normal_cdf(valeur_z) * 100
    
    print("valeur_z : \n", valeur_z)
    print("proba_actuelle : \n", proba_actuelle)

    # --- EN-TÊTE DU DASHBOARD ---
    st.title(f"⚡ Suivi Orage - {aeroport_choisi}")
    
    # --- KPI GLOBAUX DU MODÈLE POUR CET AÉROPORT ---
    st.markdown("##### 📊 Performances de l'Intelligence Artificielle sur ce site")
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("Erreur moyenne (MAE)", f"{mae_aeroport:.1f} min")
    col_kpi2.metric("Incertitude (Sigma)", f"{sigma_aeroport:.1f} min")
    col_kpi3.metric("Prédiction pour l'orage en cours", f"{prediction_modele:.1f} min")

    st.divider()

    # --- SECTION 1 : STATUT EN DIRECT (Basé sur le slider) ---
    st.markdown("### 🔴 Statut de la cellule orageuse en cours")
    
    if proba_actuelle >= 95:
        statut, couleur, message = "🟢 VERT", "green", "Alerte levée. Reprise des opérations."
    elif proba_actuelle >= 75:
        statut, couleur, message = "🟠 ORANGE", "orange", "Orage en dissipation. Préparation des équipes."
    else:
        statut, couleur, message = "🔴 ROUGE", "red", "Orage actif. Opérations suspendues."

    col1, col2, col3 = st.columns(3)
    col1.metric(label="Statut de sécurité", value=statut, delta=message, delta_color=couleur)
    col2.metric(label="Probabilité de fin d'orage", value=f"{proba_actuelle:.1f} %")
    
    # Calcul du gain par rapport à la règle des 30 minutes
    temps_standard = duree_reelle + 30
    if proba_actuelle >= 95:
        gain = int(temps_standard - temps_ecoule)
        if gain > 0:
            col3.metric(label="Temps gagné estimé", value=f"{gain} min", delta="Par rapport aux 30min fixes")
        else:
            col3.metric(label="Temps gagné estimé", value="0 min", delta="Règle fixe plus rapide ici", delta_color="inverse")
    else:
        col3.metric(label="Temps gagné estimé", value="En attente", delta="Seuil de 95% non atteint", delta_color="off")

    # --- SECTION 2 : CARTE & GRAPHIQUE ---
    col_map, col_graph = st.columns([1, 1])

    with col_map:
        st.subheader("📍 Zone de surveillance")
        lat, lon = coords_aeroports.get(aeroport_choisi, [46.0, 2.0]) # Coordonnées France par défaut
        m = folium.Map(location=[lat, lon], zoom_start=11)
        
        folium.Marker([lat, lon], popup=aeroport_choisi, icon=folium.Icon(color='blue', icon='plane')).add_to(m)
        folium.Circle(location=[lat, lon], radius=20000, color='red', fill=True, fill_opacity=0.1).add_to(m)
        st_folium(m, width=500, height=400)

    with col_graph:
        st.subheader("📈 Évolution de la probabilité")
        # Création des données pour la courbe du graphique
        minutes_x = np.linspace(0, 120, 100)
        probas_y = [normal_cdf((m - prediction_modele) / sigma_aeroport) * 100 for m in minutes_x]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(minutes_x, probas_y, color='blue', linewidth=2, label="Courbe de probabilité")
        ax.axhline(y=95, color='green', linestyle='--', label='Seuil de sécurité (95%)')
        
        # Ligne verticale rouge pour montrer où on en est avec le slider
        ax.axvline(x=temps_ecoule, color='red', linestyle='-', linewidth=2, label='Temps actuel (Curseur)')
        
        ax.set_xlabel("Minutes écoulées")
        ax.set_ylabel("Probabilité de fin (%)")
        ax.set_ylim(0, 105)
        ax.set_xlim(0, max(80, temps_ecoule + 10))
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)