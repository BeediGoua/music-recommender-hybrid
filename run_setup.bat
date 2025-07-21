@echo off
echo  === CONFIGURATION MUSIC RECOMMENDER ===
echo.

echo  Étape 1: Création des fichiers modèles...
python create_models_manual.py
echo.

echo  Étape 2: Vérification des fichiers...
if exist "data\processed\songs_metadata_clean.csv" (
    echo  songs_metadata_clean.csv créé
) else (
    echo  songs_metadata_clean.csv manquant
)

if exist "data\processed\word2vec_similarities.json" (
    echo  word2vec_similarities.json créé
) else (
    echo  word2vec_similarities.json manquant
)

if exist "data\processed\content_embeddings.npy" (
    echo  content_embeddings.npy créé
) else (
    echo  content_embeddings.npy manquant
)

echo.
echo  Configuration terminée !
echo.
echo  Prochaines étapes:
echo    1. Testez l'app: streamlit run app/streamlit_app_deployment.py
echo    2. Commitez et pushez sur GitHub
echo    3. Déployez sur Streamlit Cloud
echo.
pause