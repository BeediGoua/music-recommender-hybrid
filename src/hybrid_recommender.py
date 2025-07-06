"""
hybrid_recommender.py

Module du moteur de recommandation hybride :
- Word2Vec (co-occurrence)
- SentenceTransformer (contenu)
- Combinaison pondérée
- Outils d'affichage et vérifications
"""

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import time


def recommend_word2vec(song_id, model, topn=10):
    """
    Recommandation Word2Vec seule.
    """
    if song_id not in model.wv:
        print(f"ID {song_id} non trouvé dans vocab Word2Vec ➜ fallback Content-only")
        return []
    similar = model.wv.most_similar(positive=[song_id], topn=topn)
    return [(sid, round(score, 4)) for sid, score in similar]


def recommend_content(song_id, songs_df, embeddings, topn=10):
    """
    Recommandation Content-based seule.
    """
    idx = songs_df[songs_df['track_id'] == song_id].index[0]
    sims = cosine_similarity([embeddings[idx]], embeddings)[0]
    top_idx = sims.argsort()[::-1][1:topn+1]
    return [(songs_df.iloc[i]['track_id'], round(sims[i], 4)) for i in top_idx]


def recommend_hybrid(song_id, model, songs_df, embeddings, alpha=0.7, topn=10, same_genre=True):
    """
    Recommandation hybride.
    """
    t0 = time.time()

    sim_w2v = dict(recommend_word2vec(song_id, model, topn=50))
    if not sim_w2v:
        print("Fallback ➜ Content-only.")
        return recommend_content(song_id, songs_df, embeddings, topn)

    idx_target = songs_df[songs_df['track_id'] == song_id].index[0]
    sims_content = cosine_similarity([embeddings[idx_target]], embeddings)[0]

    w2v_scores, content_scores, sids = [], [], []
    for sid, w2v_score in sim_w2v.items():
        idx2 = songs_df[songs_df['track_id'] == sid].index[0]
        content_score = sims_content[idx2]
        sids.append(sid)
        w2v_scores.append(w2v_score)
        content_scores.append(content_score)

    scaler = MinMaxScaler()
    w2v_norm = scaler.fit_transform(np.array(w2v_scores).reshape(-1, 1)).flatten()
    content_norm = scaler.fit_transform(np.array(content_scores).reshape(-1, 1)).flatten()

    hybrid_scores = []
    for sid, w2v_s, c_s in zip(sids, w2v_norm, content_norm):
        final_score = alpha * w2v_s + (1 - alpha) * c_s
        hybrid_scores.append((sid, round(final_score, 4)))

    genre_ref = songs_df[songs_df['track_id'] == song_id]['genre'].values[0]
    if same_genre:
        hybrid_scores = [
            (sid, score) for sid, score in hybrid_scores
            if songs_df[songs_df['track_id'] == sid]['genre'].values[0] == genre_ref
        ]

    hybrid_scores = [(sid, score) for sid, score in hybrid_scores if sid != song_id]
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:topn]

    print(f"Hybrid généré en {round(time.time() - t0, 3)} sec")
    return hybrid_scores


def show_reco_detail(results, songs_df):
    out = []
    for sid, score in results:
        row = songs_df[songs_df['track_id'] == sid].iloc[0]
        out.append({
            "track_id": sid,
            "title": row['title'],
            "artist": row['artist'],
            "genre": row['genre'],
            "duration_sec": row['duration_sec'],  # ✅
            "score": score
        })
    df = pd.DataFrame(out)
    assert 'duration_sec' in df.columns, "duration_sec manquant !"
    return df


