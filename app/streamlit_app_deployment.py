"""
streamlit_app_deployment.py - Version optimisée pour le déploiement

Version adaptée qui utilise des modèles simulés pour fonctionner
sans dépendances lourdes sur Streamlit Cloud.
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import time

# Configuration de la page
st.set_page_config(
    page_title="Music Recommender - AI Portfolio Project",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Classes simulées pour remplacer les vraies
class SimulatedWord2Vec:
    """Classe qui simule un modèle Word2Vec"""
    
    def __init__(self, similarities_dict):
        self.similarities = similarities_dict
        self.wv = self
    
    def __contains__(self, item):
        return item in self.similarities
    
    def most_similar(self, positive, topn=10):
        if not positive or positive[0] not in self.similarities:
            return []
        
        song_id = positive[0]
        similarities = self.similarities.get(song_id, [])
        return similarities[:topn]

@st.cache_data
def load_data_and_models():
    """Charger toutes les données et modèles de manière optimisée"""
    try:
        # Déterminer le chemin de base
        base_dir = Path(__file__).resolve().parent.parent
        processed_dir = base_dir / "data" / "processed"
        progress_bar = st.progress(0, text="Initialisation...")

        progress_bar.progress(25, text="Chargement du catalogue musical...")
        
        
        # 1. Charger le dataset principal
        songs_df = pd.read_csv(processed_dir / "songs_metadata_clean.csv")
        
        # 2. Charger les similarités Word2Vec
        progress_bar.progress(50, text="Chargement du modèle Word2Vec...")
        with open(processed_dir / "word2vec_similarities.json", "r") as f:
            similarities = json.load(f)
        w2v_model = SimulatedWord2Vec(similarities)

        
        # 3. Charger les embeddings de contenu
        progress_bar.progress(75, text="Chargement des embeddings de contenu...")
        content_embeddings = np.load(processed_dir / "content_embeddings.npy")

        progress_bar.progress(100, text="Prêt !")
        time.sleep(0.5)
        progress_bar.empty()
        return songs_df, w2v_model, content_embeddings
    
    except Exception as e:
        st.error(f"Erreur de chargement: {e}")
        st.info("Utilisation du mode démo avec des données simulées...")
        
        # Créer des données de démonstration si le chargement échoue
        demo_data = create_demo_data()
        return demo_data

def create_premium_demo_data():
    """Créer des données de démonstration enrichies."""
    genres = ['Pop', 'Rock', 'Electronic', 'Hip-Hop', 'Jazz', 'Classical', 'R&B', 'Country', 'Reggae', 'Blues']
    n_songs = 1000
    demo_songs = {
        'track_id': [f'demo_{i}' for i in range(n_songs)],
        'title': [f'Song Title {i}' for i in range(n_songs)],
        'artist': [f'Artist {i//20}' for i in range(n_songs)],
        'genre': [random.choice(genres) for _ in range(n_songs)],
        'duration_sec': np.random.uniform(120, 360, n_songs),
        'popularity': np.random.uniform(0, 100, n_songs),
        'energy': np.random.uniform(0, 1, n_songs),
        'valence': np.random.uniform(0, 1, n_songs),
        'danceability': np.random.uniform(0, 1, n_songs),
        'acousticness': np.random.uniform(0, 1, n_songs),
        'tempo': np.random.uniform(60, 200, n_songs),
        'decade': [random.choice([1970, 1980, 1990, 2000, 2010, 2020]) for _ in range(n_songs)]
    }
    songs_df = pd.DataFrame(demo_songs)
    similarities = {}
    for i in range(n_songs):
        track_id = f'demo_{i}'
        genre = songs_df.iloc[i]['genre']
        similar_candidates = songs_df[(songs_df['genre'] == genre) & (songs_df['track_id'] != track_id)]
        if not similar_candidates.empty:
            target_features = ['energy', 'valence', 'danceability', 'acousticness']
            target_values = songs_df.iloc[i][target_features].values
            similar_tracks = []
            for _, candidate in similar_candidates.head(15).iterrows():
                candidate_values = candidate[target_features].values
                similarity = 1 - np.linalg.norm(target_values - candidate_values) / 4
                similar_tracks.append((candidate['track_id'], float(max(0.1, similarity))))
            similarities[track_id] = sorted(similar_tracks, key=lambda x: x[1], reverse=True)[:10]
    w2v_model = SimulatedWord2Vec(similarities)
    feature_matrix = songs_df[['energy', 'valence', 'danceability', 'acousticness', 'popularity', 'tempo']].values
    embeddings = np.random.rand(n_songs, 384).astype(np.float32)
    for i in range(min(6, 384)):
        embeddings[:, i] = feature_matrix[:, i % feature_matrix.shape[1]]
    return songs_df, w2v_model, embeddings


def recommend_hybrid_optimized(song_id, model, songs_df, embeddings, alpha=0.7, topn=10, same_genre=True):
    """Version optimisée de la recommandation hybride"""
    start_time = time.time()
    
    try:
        # 1. Recommandations Word2Vec
        if song_id in model:
            w2v_results = model.most_similar([song_id], topn=min(50, topn*3))
            sim_w2v = dict(w2v_results)
        else:
            sim_w2v = {}
        
        # Si pas de résultats Word2Vec, utiliser seulement le contenu
        if not sim_w2v:
            return recommend_content_only(song_id, songs_df, embeddings, topn, same_genre)
        
        # 2. Recommandations basées sur le contenu
        try:
            idx_target = songs_df[songs_df['track_id'] == song_id].index[0]
            target_embedding = embeddings[idx_target].reshape(1, -1)
            similarities_content = cosine_similarity(target_embedding, embeddings)[0]
        except IndexError:
            # Si le morceau n'est pas trouvé, retourner les top Word2Vec
            return [(sid, score) for sid, score in list(sim_w2v.items())[:topn]]
        
        # 3. Fusion hybride
        hybrid_scores = []
        
        for sid, w2v_score in sim_w2v.items():
            try:
                idx_sim = songs_df[songs_df['track_id'] == sid].index[0]
                content_score = similarities_content[idx_sim]
                
                # Score hybride pondéré
                hybrid_score = alpha * w2v_score + (1 - alpha) * content_score
                hybrid_scores.append((sid, float(hybrid_score)))
            except IndexError:
                # Si le morceau similaire n'est pas trouvé, utiliser seulement Word2Vec
                hybrid_scores.append((sid, float(w2v_score * alpha)))
        
        # 4. Filtrage par genre si demandé
        if same_genre:
            try:
                target_genre = songs_df[songs_df['track_id'] == song_id]['genre'].iloc[0]
                filtered_scores = []
                for sid, score in hybrid_scores:
                    try:
                        song_genre = songs_df[songs_df['track_id'] == sid]['genre'].iloc[0]
                        if song_genre == target_genre:
                            filtered_scores.append((sid, score))
                    except IndexError:
                        continue
                hybrid_scores = filtered_scores
            except IndexError:
                pass  # Garder tous les résultats si genre non trouvé
        
        # 5. Tri et limitation
        hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
        hybrid_scores = hybrid_scores[:topn]
        
        # Temps de traitement
        processing_time = time.time() - start_time
        
        return hybrid_scores, processing_time
    
    except Exception as e:
        st.error(f"Erreur dans recommend_hybrid: {e}")
        return [], 0

def recommend_content_only(song_id, songs_df, embeddings, topn=10, same_genre=True):
    """Recommandation basée uniquement sur le contenu"""
    try:
        idx = songs_df[songs_df['track_id'] == song_id].index[0]
        target_embedding = embeddings[idx].reshape(1, -1)
        similarities = cosine_similarity(target_embedding, embeddings)[0]
        
        # Obtenir les indices triés par similarité
        top_indices = similarities.argsort()[::-1][1:topn+1]  # Exclure le morceau lui-même
        
        results = []
        for idx in top_indices:
            sid = songs_df.iloc[idx]['track_id']
            score = float(similarities[idx])
            results.append((sid, score))
        
        return results
    except IndexError:
        return []

def show_recommendation_details(results, songs_df):
    """Afficher les détails des recommandations"""
    if not results:
        return pd.DataFrame()
    
    # Si results contient le temps de traitement
    if isinstance(results, tuple):
        recommendations, processing_time = results
        st.info(f"Traité en {processing_time:.3f} secondes")
    else:
        recommendations = results
    
    details = []
    for sid, score in recommendations:
        try:
            row = songs_df[songs_df['track_id'] == sid].iloc[0]
            details.append({
                "Titre": row['title'],
                "Artiste": row['artist'],
                "Genre": row['genre'],
                "⏱Durée (s)": f"{row['duration_sec']:.0f}",
                "Score": f"{score:.3f}"
            })
        except IndexError:
            # Morceau non trouvé, ignorer
            continue
    
    return pd.DataFrame(details)

# Interface utilisateur
def main():
    """Interface principale de l'application"""
    
    # CSS personnalisé
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        .stDataFrame {
            border: 1px solid #ddd;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎵 Music Recommender</h1>
        <p>Système de recommandation hybride - <b>Word2Vec + NLP</b></p>
        <p><i>Projet Portfolio de Data Science</i></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Charger les données
    with st.spinner("Chargement des modèles..."):
        songs_df, w2v_model, content_embeddings = load_data_and_models()
    
    if songs_df is None:
        st.error("Impossible de charger les données.")
        return
    
    # Métriques du projet
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Morceaux", f"{len(songs_df):,}")
    with col2:
        st.metric("Genres", f"{songs_df['genre'].nunique()}")
    with col3:
        st.metric("Artistes", f"{songs_df['artist'].nunique()}")
    with col4:
        st.metric("Algorithme", "Hybride")
    
    st.markdown("---")
    
    # Interface principale
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.header("Paramètres")
        
        # Recherche de morceaux
        search_query = st.text_input(
            "Rechercher un morceau:",
            placeholder="Tapez un titre ou un artiste..."
        )
        
        # Filtrer les options
        if search_query:
            mask = (
                songs_df['title'].str.contains(search_query, case=False, na=False) |
                songs_df['artist'].str.contains(search_query, case=False, na=False)
            )
            filtered_df = songs_df[mask]
        else:
            filtered_df = songs_df.head(100)  # Limiter pour les performances
        
        # Sélection du morceau
        if len(filtered_df) > 0:
            options = filtered_df.apply(
                lambda x: f"{x['title']} - {x['artist']} ({x['genre']})", 
                axis=1
            )
            
            selected_display = st.selectbox(
                "🎵 Choisir un morceau:", 
                options,
                key="song_selector"
            )
            
            # Obtenir l'ID du morceau sélectionné
            selected_idx = options[options == selected_display].index[0]
            selected_id = songs_df.loc[selected_idx, 'track_id']
            
            # Paramètres de recommandation
            st.subheader("Configuration")
            
            alpha = st.slider(
                "Pondération Word2Vec vs Contenu",
                0.0, 1.0, 0.7, 0.05,
                help="0 = 100% contenu, 1 = 100% collaboratif"
            )
            
            topn = st.slider("Nombre de recommandations", 1, 20, 10)
            
            same_genre = st.checkbox(
                "Même genre uniquement", 
                value=True,
                help="Limiter aux morceaux du même genre"
            )
            
            # Filtres additionnels
            if 'duration_sec' in songs_df.columns:
                duration_range = st.slider(
                    "Durée (secondes)",
                    int(songs_df['duration_sec'].min()),
                    int(songs_df['duration_sec'].max()),
                    (60, 300)
                )
            else:
                duration_range = None
            
            # Bouton de recommandation
            if st.button("Générer les Recommandations", type="primary"):
                
                with st.spinner("Calcul des similarités..."):
                    # Générer les recommandations
                    results = recommend_hybrid_optimized(
                        selected_id, w2v_model, songs_df, content_embeddings,
                        alpha=alpha, topn=topn, same_genre=same_genre
                    )
                    
                    if results:
                        st.session_state['recommendations'] = results
                        st.session_state['selected_song'] = {
                            'id': selected_id,
                            'title': songs_df.loc[selected_idx, 'title'],
                            'artist': songs_df.loc[selected_idx, 'artist'],
                            'genre': songs_df.loc[selected_idx, 'genre']
                        }
                        st.success("Recommandations générées!")
                    else:
                        st.warning("Aucune recommandation trouvée.")
        else:
            st.warning("Aucun morceau trouvé pour votre recherche.")
    
    with col_right:
        st.header("🎵 Recommandations")
        
        # Affichage du morceau sélectionné
        if 'selected_song' in st.session_state:
            selected = st.session_state['selected_song']
            st.info(f"**Basé sur:** {selected['title']} - {selected['artist']} ({selected['genre']})")
        
        # Affichage des recommandations
        if 'recommendations' in st.session_state:
            results = st.session_state['recommendations']
            details_df = show_recommendation_details(results, songs_df)
            
            if not details_df.empty:
                # Filtrer par durée si spécifié
                if duration_range and 'duration_sec' in songs_df.columns:
                    # Cette logique nécessiterait plus de travail pour être complètement fonctionnelle
                    pass
                
                st.dataframe(
                    details_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Bouton de téléchargement
                csv_data = details_df.to_csv(index=False)
                st.download_button(
                    "Télécharger les résultats (CSV)",
                    data=csv_data,
                    file_name="music_recommendations.csv",
                    mime="text/csv"
                )
                
                # Statistiques des recommandations
                if len(details_df) > 0:
                    st.subheader("Statistiques")
                    genre_counts = pd.Series([row.split('(')[-1].replace(')', '') for row in details_df['🎪 Genre']]).value_counts()
                    st.bar_chart(genre_counts)
            else:
                st.info("Aucune recommandation à afficher.")
        else:
            st.info("Sélectionnez un morceau et cliquez sur 'Générer les Recommandations' pour commencer!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🎵 <b>Music Recommender Hybrid System</b> - Projet Portfolio Data Science</p>
        <p><i>Powered by Streamlit • Word2Vec + NLP • Beedi GOUA</i></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()