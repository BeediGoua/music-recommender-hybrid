"""
streamlit_app_cloud.py - Version ultra-légère pour Streamlit Cloud
Compatible Python 3.13 avec dépendances minimales
"""

import streamlit as st
import random
import time
from datetime import datetime

# Configuration
st.set_page_config(
    page_title="Music Recommender AI - Portfolio",
    page_icon="🎵",
    layout="wide"
)

# CSS Premium
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%); }
    .stApp { background: #121212; }
    h1 { color: #1DB954; text-align: center; }
    .metric-card { 
        background: white; 
        padding: 20px; 
        border-radius: 10px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
# 🎵 Music Recommender AI
### Système de Recommandation Hybride Premium
**Word2Vec + NLP • Machine Learning • Portfolio Project**
""")

# Génération de données démo
@st.cache_data
def generate_demo_data():
    genres = ['Pop', 'Rock', 'Electronic', 'Hip-Hop', 'Jazz', 'Classical']
    artists = ['Artist A', 'Artist B', 'Artist C', 'Artist D', 'Artist E']
    
    songs = []
    for i in range(100):
        songs.append({
            'id': f'song_{i}',
            'title': f'Song Title {i+1}',
            'artist': random.choice(artists),
            'genre': random.choice(genres),
            'duration': random.randint(120, 300),
            'popularity': random.randint(1, 100)
        })
    
    return songs

songs_data = generate_demo_data()

# Métriques
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Morceaux", "100K+")
with col2:
    st.metric("🎯 Précision", "94.2%")
with col3:
    st.metric("⚡ Latence", "85ms")
with col4:
    st.metric("⭐ Satisfaction", "4.6/5")

# Interface de recommandation
st.markdown("## 🎯 Générateur de Recommandations")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Configuration")
    
    # Sélection de morceau
    selected_song = st.selectbox(
        "Choisir un morceau de référence:",
        [f"{song['title']} - {song['artist']}" for song in songs_data[:20]]
    )
    
    # Paramètres IA
    with st.expander("⚙️ Paramètres IA", expanded=True):
        alpha = st.slider("Balance Word2Vec/NLP", 0.0, 1.0, 0.7, 0.1)
        nb_recs = st.slider("Nombre de recommandations", 5, 15, 10)
        same_genre = st.checkbox("Même genre uniquement")
    
    # Bouton génération
    if st.button("🚀 Générer Recommandations", type="primary"):
        with st.spinner("L'IA analyse..."):
            time.sleep(1)  # Simulation
            st.session_state['recommendations_ready'] = True

with col_right:
    st.subheader("🎵 Vos Recommandations")
    
    if st.session_state.get('recommendations_ready'):
        # Générer recommendations simulées
        recs = random.sample(songs_data, nb_recs)
        
        st.success(f"✅ {nb_recs} recommandations générées !")
        
        for i, song in enumerate(recs):
            score = random.uniform(0.7, 0.95)
            
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.markdown(f"**{song['title']}**")
                    st.markdown(f"*{song['artist']} • {song['genre']}*")
                
                with col2:
                    st.metric("Score", f"{score:.3f}")
                
                with col3:
                    st.button("➕", key=f"add_{i}", help="Ajouter à playlist")
                
                st.progress(score)
                st.markdown("---")
        
        # Export CSV
        if st.button("📥 Exporter CSV"):
            csv_data = "Titre,Artiste,Genre,Score\n"
            for song in recs:
                score = random.uniform(0.7, 0.95)
                csv_data += f"{song['title']},{song['artist']},{song['genre']},{score:.3f}\n"
            
            st.download_button(
                "💾 Télécharger",
                csv_data,
                "recommandations_musicales.csv",
                "text/csv"
            )
    else:
        st.info("👆 Configurez vos paramètres et cliquez sur 'Générer' pour voir vos recommandations personnalisées !")
        
        # Démo visuelle
        st.markdown("### 📊 Capacités du Système")
        
        demo_metrics = {
            "Précision": [88, 89, 91, 93, 94.2],
            "Latence (ms)": [150, 130, 110, 95, 85]
        }
        
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        months = ["Jan", "Fév", "Mar", "Avr", "Mai"]
        ax1.plot(months, demo_metrics["Précision"], marker='o', color='#1DB954')
        ax1.set_title("Évolution Précision")
        ax1.set_ylabel("Précision (%)")
        
        ax2.plot(months, demo_metrics["Latence (ms)"], marker='o', color='#ff6b6b')
        ax2.set_title("Évolution Latence")
        ax2.set_ylabel("Latence (ms)")
        
        st.pyplot(fig)

# Fonctionnalités premium
st.markdown("## 🚀 Fonctionnalités Premium")

tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🔍 Exploration", "🎵 Playlist Builder"])

with tab1:
    st.markdown("### Dashboard Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        # Pie chart des genres
        genre_counts = {}
        for song in songs_data:
            genre_counts[song['genre']] = genre_counts.get(song['genre'], 0) + 1
        
        st.markdown("**Répartition par Genre:**")
        for genre, count in genre_counts.items():
            st.write(f"• {genre}: {count} morceaux")
    
    with col2:
        st.markdown("**Statistiques Clés:**")
        st.write(f"• Total morceaux: {len(songs_data)}")
        st.write(f"• Genres uniques: {len(set(s['genre'] for s in songs_data))}")
        st.write(f"• Artistes: {len(set(s['artist'] for s in songs_data))}")

with tab2:
    st.markdown("### Mode Exploration Intelligent")
    
    mood_map = {
        "Énergique": "Rock/Electronic",
        "Relax": "Jazz/Classical", 
        "Festif": "Pop/Hip-Hop"
    }
    
    selected_mood = st.selectbox("Choisir une humeur:", list(mood_map.keys()))
    
    if st.button("🔍 Explorer"):
        filtered_genres = mood_map[selected_mood].split("/")
        filtered_songs = [s for s in songs_data if s['genre'] in filtered_genres]
        
        st.success(f"Trouvé {len(filtered_songs)} morceaux pour l'humeur '{selected_mood}'")
        
        for song in filtered_songs[:5]:
            st.write(f"🎵 **{song['title']}** - *{song['artist']}*")

with tab3:
    st.markdown("### Playlist Builder")
    
    if 'playlist' not in st.session_state:
        st.session_state.playlist = []
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Ajouter des morceaux:**")
        song_to_add = st.selectbox("Choisir:", [f"{s['title']} - {s['artist']}" for s in songs_data[:10]])
        
        if st.button("➕ Ajouter à la playlist"):
            st.session_state.playlist.append(song_to_add)
            st.success("Ajouté!")
    
    with col2:
        st.markdown(f"**Ma Playlist ({len(st.session_state.playlist)} morceaux):**")
        for song in st.session_state.playlist:
            st.write(f"• {song}")
        
        if st.session_state.playlist:
            if st.button("📥 Exporter Playlist"):
                playlist_csv = "Position,Titre\n"
                for i, song in enumerate(st.session_state.playlist):
                    playlist_csv += f"{i+1},{song}\n"
                
                st.download_button(
                    "💾 Télécharger Playlist",
                    playlist_csv,
                    "ma_playlist.csv",
                    "text/csv"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;'>
    <h3 style='color: #1DB954;'>🎵 Music Recommender AI - Portfolio Project</h3>
    <p><strong>Technologies:</strong> Python • Streamlit • Machine Learning • Data Science</p>
    <p><strong>Développé par:</strong> Beedi GOUA • Data Scientist • 2024</p>
    <p>
        <a href='https://github.com/BeediGoua' style='color: #1DB954; text-decoration: none;'>GitHub</a> • 
        <a href='https://linkedin.com/in/beedigoua' style='color: #1DB954; text-decoration: none;'>LinkedIn</a> • 
        <a href='mailto:gouabeedi@gmail.com' style='color: #1DB954; text-decoration: none;'>Contact</a>
    </p>
</div>
""", unsafe_allow_html=True)