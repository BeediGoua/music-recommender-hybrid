"""
preprocess_for_deployment.py - Préparation des données pour Streamlit Cloud

Script optimisé pour préparer un dataset léger et efficace pour le déploiement.
Gère les contraintes de mémoire et de taille de Streamlit Cloud.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def check_file_sizes():
    """Vérifier les tailles de fichiers pour Streamlit Cloud"""
    print("Vérification des tailles de fichiers...")
    
    files_to_check = [
        "data/SpotifyFeatures.csv",
        "data/processed/songs_metadata_clean.csv",
        "data/processed/word2vec.model",
        "data/processed/content_embeddings.npy"
    ]
    
    total_size = 0
    for file_path in files_to_check:
        if Path(file_path).exists():
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            print(f"  {file_path}: {size_mb:.2f} MB")
            total_size += size_mb
        else:
            print(f"  {file_path}: Non trouvé")
    
    print(f"\nTaille totale: {total_size:.2f} MB")
    
    # Limite recommandée pour Streamlit Cloud
    if total_size > 800:
        print(" Attention: Taille > 800MB, réduction recommandée")
        return False
    else:
        print(" Taille acceptable pour Streamlit Cloud")
        return True

def optimize_dataset_for_deployment():
    """Optimiser le dataset pour un déploiement efficace"""
    print(" Optimisation du dataset pour le déploiement...")
    
    # Charger les données
    data_path = Path("data/SpotifyFeatures.csv")
    if not data_path.exists():
        print(" Fichier data/SpotifyFeatures.csv non trouvé!")
        return None
    
    df = pd.read_csv(data_path)
    print(f" Dataset original: {len(df)} morceaux")
    
    # Nettoyer et optimiser
    df_clean = prepare_clean_dataset(df)
    
    # Stratégie d'échantillonnage intelligent
    df_optimized = smart_sampling(df_clean)
    
    # Sauvegarder
    output_path = Path("data/processed/songs_metadata_clean.csv")
    df_optimized.to_csv(output_path, index=False)
    
    print(f" Dataset optimisé sauvegardé: {len(df_optimized)} morceaux")
    print(f" Fichier: {output_path}")
    
    return df_optimized

def prepare_clean_dataset(df):
    """Nettoyer et préparer le dataset"""
    print(" Nettoyage des données...")
    
    # Renommer les colonnes
    column_mapping = {
        'track_name': 'title',
        'artist_name': 'artist',
        'duration_ms': 'duration_sec'
    }
    df_clean = df.rename(columns=column_mapping)
    
    # Convertir durée ms -> secondes
    if 'duration_sec' in df_clean.columns:
        df_clean['duration_sec'] = df_clean['duration_sec'] / 1000
    
    # Filtres qualité
    df_clean = df_clean[
        (df_clean['duration_sec'] >= 30) &      # Minimum 30s
        (df_clean['duration_sec'] <= 600) &     # Maximum 10min
        (df_clean['title'].notna()) &           # Titre non vide
        (df_clean['artist'].notna()) &          # Artiste non vide
        (df_clean['genre'].notna())             # Genre non vide
    ]
    
    # Supprimer doublons
    df_clean = df_clean.drop_duplicates(subset=['title', 'artist'])
    
    # Créer track_id si manquant
    if 'track_id' not in df_clean.columns:
        df_clean['track_id'] = 'track_' + df_clean.index.astype(str)
    
    print(f" Après nettoyage: {len(df_clean)} morceaux")
    return df_clean

def smart_sampling(df, target_size=8000):
    """Échantillonnage intelligent pour préserver la diversité"""
    print(f" Échantillonnage intelligent (cible: {target_size} morceaux)...")
    
    if len(df) <= target_size:
        print(" Dataset déjà de taille optimale")
        return df
    
    # Stratégie: garder une représentation équilibrée des genres
    samples = []
    
    # Calculer le nombre de morceaux par genre
    genre_counts = df['genre'].value_counts()
    total_genres = len(genre_counts)
    
    # Répartition proportionnelle mais avec minimum par genre
    min_per_genre = max(10, target_size // (total_genres * 2))  # Min 10 par genre
    
    for genre in genre_counts.index:
        genre_data = df[df['genre'] == genre]
        
        # Nombre de morceaux à prendre pour ce genre
        genre_target = min(
            len(genre_data),
            max(min_per_genre, int(target_size * len(genre_data) / len(df)))
        )
        
        # Échantillonnage stratifié (populaires + aléatoires)
        if 'popularity' in genre_data.columns:
            # Prendre 30% des plus populaires + 70% aléatoires
            n_popular = int(genre_target * 0.3)
            n_random = genre_target - n_popular
            
            popular = genre_data.nlargest(n_popular, 'popularity')
            remaining = genre_data.drop(popular.index)
            if len(remaining) > 0:
                random_sample = remaining.sample(n=min(n_random, len(remaining)), random_state=42)
                genre_sample = pd.concat([popular, random_sample])
            else:
                genre_sample = popular
        else:
            # Échantillonnage aléatoire simple
            genre_sample = genre_data.sample(n=genre_target, random_state=42)
        
        samples.append(genre_sample)
        print(f"  🎵 {genre}: {len(genre_sample)} morceaux")
    
    # Combiner tous les échantillons
    df_sampled = pd.concat(samples, ignore_index=True)
    
    # Si encore trop gros, échantillonnage final aléatoire
    if len(df_sampled) > target_size:
        df_sampled = df_sampled.sample(n=target_size, random_state=42)
    
    print(f" Échantillonnage terminé: {len(df_sampled)} morceaux")
    return df_sampled

def create_minimal_test_dataset():
    """Créer un dataset minimal pour tests rapides"""
    print(" Création d'un dataset minimal pour tests...")
    
    # Dataset de test avec genres populaires
    test_data = {
        'track_id': [f'test_{i}' for i in range(100)],
        'title': [f'Test Song {i}' for i in range(100)],
        'artist': [f'Artist {i//10}' for i in range(100)],
        'genre': ['Pop', 'Rock', 'Electronic', 'Hip-Hop', 'Jazz'] * 20,
        'duration_sec': np.random.uniform(120, 300, 100),
        'popularity': np.random.randint(0, 100, 100),
        'energy': np.random.uniform(0, 1, 100),
        'valence': np.random.uniform(0, 1, 100)
    }
    
    df_test = pd.DataFrame(test_data)
    
    # Sauvegarder
    test_path = Path("data/processed/songs_metadata_test.csv")
    df_test.to_csv(test_path, index=False)
    
    print(f" Dataset de test créé: {test_path}")
    return df_test

def validate_processed_data():
    """Valider les données traitées"""
    print(" Validation des données traitées...")
    
    required_files = [
        "data/processed/songs_metadata_clean.csv",
        "data/processed/word2vec.model",
        "data/processed/content_embeddings.npy"
    ]
    
    all_valid = True
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   {file_path}: OK")
        else:
            print(f"   {file_path}: MANQUANT")
            all_valid = False
    
    if all_valid:
        print("\n Tous les fichiers requis sont présents!")
        
        # Vérifier la cohérence des données
        df = pd.read_csv("data/processed/songs_metadata_clean.csv")
        embeddings = np.load("data/processed/content_embeddings.npy")
        
        print(f" Morceaux: {len(df)}")
        print(f" Embeddings: {embeddings.shape}")
        
        if len(df) == embeddings.shape[0]:
            print(" Cohérence données/embeddings: OK")
        else:
            print("  Incohérence entre données et embeddings!")
            all_valid = False
    
    return all_valid

def main():
    """Fonction principale de préparation"""
    print(" === PRÉPARATION DONNÉES POUR DÉPLOIEMENT ===\n")
    
    try:
        # 1. Vérifier l'état actuel
        check_file_sizes()
        print()
        
        # 2. Optimiser le dataset
        df = optimize_dataset_for_deployment()
        if df is None:
            print(" Impossible de charger les données")
            return 1
        print()
        
        # 3. Créer dataset de test si nécessaire
        create_minimal_test_dataset()
        print()
        
        # 4. Valider le résultat
        if validate_processed_data():
            print(" Préparation terminée avec succès!")
        else:
            print("  Préparation terminée avec des avertissements")
        
        print("\n Prochaines étapes:")
        print("   1. Exécutez: python setup.py")
        print("   2. Testez: streamlit run app/streamlit_app.py")
        print("   3. Déployez sur Streamlit Cloud")
        
        return 0
        
    except Exception as e:
        print(f"\n ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())