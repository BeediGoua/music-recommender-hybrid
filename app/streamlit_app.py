import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from src.hybrid_recommender import recommend_hybrid, show_reco_detail


# Chargement 

base_dir = Path(__file__).resolve().parent
while not (base_dir / "data").exists():
    base_dir = base_dir.parent

inputs_dir = base_dir / "data" / "processed"

model_path = inputs_dir / "word2vec.model"
if not model_path.exists():
    st.error(f"Modèle Word2Vec introuvable : {model_path}")
    st.stop()
w2v_model = Word2Vec.load(str(model_path))

content_embeddings = np.load(inputs_dir / "content_embeddings.npy")
songs_df = pd.read_csv(inputs_dir / "songs_metadata_clean.csv")

expected_cols = {'track_id', 'title', 'artist', 'genre', 'duration_sec'}
if not expected_cols.issubset(songs_df.columns):
    st.error(f"Colonnes manquantes dans songs_df ! Attendues : {expected_cols}")
    st.stop()


# Config + Hero + Style global


st.set_page_config(page_title="Music Recommender", layout="wide")


st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #FFFFFF;
    }
    .hero {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        padding: 3rem;
        border-radius: 10px;
        text-align: center;
        color: #FFFFFF;
    }
    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .hero p {
        font-size: 1.2rem;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background-color: #1DB954;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
    }
    .stSlider > div {
        color: white;
    }
    .stDataFrame {
        border: 1px solid #333;
    }
    footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background: #121212;
        color: #aaa;
        text-align: center;
        padding: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div class="hero">
        <h1>🎵 Music Recommender</h1>
        <p>Découvrez vos nouvelles chansons préférées grâce à un moteur hybride <b>Word2Vec + NLP</b></p>
        
    </div>
""", unsafe_allow_html=True)


# Layout principal

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Vos paramètres")

    search = st.text_input("Recherche titre ou artiste :", placeholder="Tapez un mot clé...")
    options = songs_df.apply(lambda x: f"{x['title']} - {x['artist']}", axis=1)
    if search:
        options = options[options.str.contains(search, case=False, na=False)]

    if options.empty:
        st.warning("Aucun résultat pour cette recherche.")
        st.stop()

    choice = st.selectbox("Choisir un morceau :", options)
    selected_id = songs_df.iloc[options[options == choice].index[0]]['track_id']

    alpha = st.slider("Pondération Word2Vec vs Content", 0.0, 1.0, 0.7, 0.05)
    topn = st.slider("Nombre de suggestions", 1, 20, 10)
    same_genre = st.checkbox("Même genre uniquement", value=True)

    duration_range = st.slider(
        "Plage de durée (sec)",
        int(songs_df['duration_sec'].min()),
        int(songs_df['duration_sec'].max()),
        (60, 300)
    )

    if st.button("Lancer la recommandation"):
        with st.spinner("Calcul en cours..."):
            results = recommend_hybrid(
                selected_id,
                w2v_model,
                songs_df,
                content_embeddings,
                alpha=alpha,
                topn=topn,
                same_genre=same_genre
            )

            detail = show_reco_detail(results, songs_df)

            if detail.empty:
                st.warning("Aucune recommandation trouvée pour ces paramètres.")
                st.stop()

            if 'duration_sec' not in detail.columns:
                detail = detail.merge(
                    songs_df[['track_id', 'duration_sec']],
                    on='track_id',
                    how='left'
                )

            if detail['duration_sec'].isnull().any():
                st.error("Valeurs manquantes pour duration_sec après merge.")
                st.write(detail)
                st.stop()

            detail = detail[
                (detail['duration_sec'] >= duration_range[0]) &
                (detail['duration_sec'] <= duration_range[1])
            ]

            sort_by = st.selectbox("Trier par :", ["score", "duration_sec"])
            detail = detail.sort_values(by=sort_by, ascending=False)

            st.session_state['detail'] = detail

        st.success("Recommandation prête !")

        st.download_button(
            "Télécharger CSV",
            data=detail.to_csv(index=False).encode('utf-8'),
            file_name="recommendations.csv",
            mime='text/csv'
        )

with col2:
    st.header("Vos recommandations")

    if 'detail' in st.session_state and not st.session_state['detail'].empty:
        st.dataframe(
            st.session_state['detail'],
            use_container_width=True
        )
    else:
        st.info("Lance une recommandation pour voir le tableau ici !")


# Footer 
st.markdown("""
    <footer>
        Projet Data Science — <b>Music Recommender Hybrid System</b> • Powered by <i>Streamlit</i>
    </footer>
""", unsafe_allow_html=True)
