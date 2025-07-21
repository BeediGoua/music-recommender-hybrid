"""
Script manuel pour créer les fichiers modèles nécessaires
À exécuter une seule fois avant le déploiement
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import sys

def create_songs_metadata():
    """Créer le fichier songs_metadata_clean.csv"""
    print("Création du fichier songs_metadata_clean.csv...")
    
    # Lire le dataset original
    df = pd.read_csv("data/SpotifyFeatures.csv")
    
    # Nettoyer et renommer les colonnes
    df_clean = df.rename(columns={
        'track_name': 'title',
        'artist_name': 'artist'
    })
    
    # Convertir durée ms -> seconds (la colonne s'appelle duration_ms dans le CSV)
    if 'duration_ms' in df_clean.columns:
        df_clean['duration_sec'] = df_clean['duration_ms'] / 1000
    else:
        print("⚠️  Colonne duration_ms non trouvée, utilisation de valeurs par défaut")
        df_clean['duration_sec'] = np.random.uniform(120, 300, len(df_clean))
    
    # Filtrer les durées normales
    df_clean = df_clean[
        (df_clean['duration_sec'] >= 30) & 
        (df_clean['duration_sec'] <= 600)
    ]
    
    # Supprimer les doublons
    df_clean = df_clean.drop_duplicates(subset=['title', 'artist'])
    
    # Garder seulement les colonnes nécessaires
    columns_needed = ['track_id', 'title', 'artist', 'genre', 'duration_sec']
    df_clean = df_clean[columns_needed]
    
    # Échantillonner pour le déploiement (8000 morceaux max)
    if len(df_clean) > 8000:
        # Échantillonnage stratifié par genre
        sample_dfs = []
        for genre in df_clean['genre'].unique():
            genre_df = df_clean[df_clean['genre'] == genre]
            n_samples = min(len(genre_df), max(10, int(8000 * len(genre_df) / len(df_clean))))
            sample_df = genre_df.sample(n=n_samples, random_state=42)
            sample_dfs.append(sample_df)
        
        df_clean = pd.concat(sample_dfs, ignore_index=True)
    
    # Sauvegarder
    df_clean.to_csv("data/processed/songs_metadata_clean.csv", index=False)
    print(f"Créé songs_metadata_clean.csv avec {len(df_clean)} morceaux")
    
    return df_clean

def create_word2vec_model_simulation():
    """Créer un fichier de simulation Word2Vec"""
    print("Création simulation modèle Word2Vec...")
    
    # Lire le dataset
    df = pd.read_csv("data/processed/songs_metadata_clean.csv")
    
    # Créer un dictionnaire de similarité simple basé sur les genres
    similarity_dict = {}
    
    for _, row in df.iterrows():
        track_id = row['track_id']
        genre = row['genre']
        
        # Trouver des morceaux similaires du même genre
        similar_tracks = df[df['genre'] == genre]['track_id'].tolist()
        similar_tracks = [t for t in similar_tracks if t != track_id][:10]  # Max 10 similaires
        
        # Créer scores de similarité simulés
        similarities = []
        for i, sim_track in enumerate(similar_tracks):
            score = 0.9 - (i * 0.05)  # Décroissant de 0.9 à 0.4
            similarities.append((sim_track, float(score)))
        
        similarity_dict[track_id] = similarities
    
    # Sauvegarder comme fichier JSON (plus simple que le format Word2Vec)
    with open("data/processed/word2vec_similarities.json", "w") as f:
        json.dump(similarity_dict, f, indent=2)
    
    # Créer aussi un fichier de métadonnées
    model_info = {
        "model_type": "simulated_word2vec",
        "vocab_size": len(df),
        "vector_size": 100,
        "genres": df['genre'].unique().tolist()
    }
    
    with open("data/processed/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print("Créé simulation Word2Vec (word2vec_similarities.json)")
    return similarity_dict

def create_content_embeddings():
    """Créer embeddings de contenu factices mais réalistes"""
    print("Création des embeddings de contenu...")
    
    df = pd.read_csv("data/processed/songs_metadata_clean.csv")
    n_songs = len(df)
    
    # Créer embeddings basés sur les genres et caractéristiques
    embeddings = []
    
    # Créer un mapping genre -> vecteur de base
    genres = df['genre'].unique()
    genre_vectors = {}
    
    np.random.seed(42)  # Pour la reproductibilité
    
    for i, genre in enumerate(genres):
        # Créer un vecteur de base pour chaque genre
        base_vector = np.random.normal(0, 1, 384)
        base_vector[i % 384] += 2  # Signature unique pour le genre
        genre_vectors[genre] = base_vector
    
    # Créer embeddings pour chaque morceau
    for _, row in df.iterrows():
        genre = row['genre']
        base_vec = genre_vectors[genre].copy()
        
        # Ajouter du bruit pour la variabilité
        noise = np.random.normal(0, 0.3, 384)
        embedding = base_vec + noise
        
        # Normaliser
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Sauvegarder
    np.save("data/processed/content_embeddings.npy", embeddings)
    print(f"Créé content_embeddings.npy avec forme {embeddings.shape}")
    
    return embeddings

def create_fake_word2vec_model():
    """Créer un fichier word2vec.model factice"""
    print("Création d'un fichier Word2Vec factice...")
    
    # Créer un objet simple qui simule un modèle Word2Vec
    fake_model_data = {
        "vocab": [],
        "vectors": {},
        "model_info": {
            "vector_size": 100,
            "window": 5,
            "min_count": 2,
            "workers": 4,
            "epochs": 10
        }
    }
    
    df = pd.read_csv("data/processed/songs_metadata_clean.csv")
    
    # Ajouter tous les track_ids au vocabulaire
    for track_id in df['track_id'].tolist():
        fake_model_data["vocab"].append(track_id)
        # Créer un vecteur aléatoire pour chaque track
        vector = np.random.normal(0, 1, 100).astype(np.float32)
        fake_model_data["vectors"][track_id] = vector.tolist()
    
    # Sauvegarder comme pickle
    with open("data/processed/fake_word2vec_model.pkl", "wb") as f:
        pickle.dump(fake_model_data, f)
    
    print("Créé fake_word2vec_model.pkl")
    return fake_model_data

def main():
    """Fonction principale"""
    print("=== CRÉATION MANUELLE DES MODÈLES ===\n")
    
    try:
        # 1. Créer le dataset nettoyé
        df = create_songs_metadata()
        print()
        
        # 2. Créer la simulation Word2Vec
        create_word2vec_model_simulation()
        print()
        
        # 3. Créer les embeddings de contenu
        create_content_embeddings()
        print()
        
        # 4. Créer un modèle Word2Vec factice
        create_fake_word2vec_model()
        print()
        
        print("=== TOUS LES FICHIERS CRÉÉS AVEC SUCCÈS! ===")
        print("\nFichiers créés:")
        print("    data/processed/songs_metadata_clean.csv")
        print("    data/processed/word2vec_similarities.json")
        print("    data/processed/content_embeddings.npy")
        print("    data/processed/fake_word2vec_model.pkl")
        print("    data/processed/model_info.json")
        
        print("\n  IMPORTANT: Vous devrez modifier app/streamlit_app.py")
        print("   pour utiliser ces fichiers simulés au lieu des vrais modèles.")
        
        return True
        
    except Exception as e:
        print(f"\n ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nExécutez ce script avec: python create_models_manual.py")