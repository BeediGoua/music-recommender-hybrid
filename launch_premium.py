#!/usr/bin/env python3
"""
launch_premium.py - Script de Lancement Automatique de l'Application Premium

Ce script configure automatiquement et lance l'application premium avec toutes 
les fonctionnalités différenciantes pour un déploiement portfolio parfait.
"""

import os
import sys
import subprocess
from pathlib import Path
import time

def print_header():
    """Afficher le header d'accueil"""
    print("=" * 60)
    print("🎵 MUSIC RECOMMENDER AI - PREMIUM LAUNCHER 🎵")
    print("=" * 60)
    print("Configuration automatique pour déploiement portfolio")
    print("Avec Dashboard Analytics, Exploration IA, et Playlist Builder")
    print("=" * 60)
    print()

def check_requirements():
    """Vérifier les prérequis"""
    print("Vérification des prérequis...")
    
    # Vérifier Python
    try:
        python_version = sys.version_info
        if python_version.major == 3 and python_version.minor >= 8:
            print(f"Python {python_version.major}.{python_version.minor} détecté")
        else:
            print(f"Python 3.8+ requis, version {python_version.major}.{python_version.minor} détectée")
            return False
    except:
        print("Python non détecté")
        return False
    
    # Vérifier les packages essentiels
    required_packages = ['streamlit', 'pandas', 'numpy', 'plotly']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"{package} disponible")
        except ImportError:
            print(f"{package} manquant")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nInstallation des packages manquants...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("Packages installés avec succès")
        except subprocess.CalledProcessError:
            print("Erreur lors de l'installation des packages")
            return False
    
    print()
    return True

def setup_environment():
    """Configurer l'environnement"""
    print("Configuration de l'environnement...")
    
    # Créer les dossiers nécessaires
    required_dirs = [
        "data/processed",
        "outputs",
        ".streamlit"
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"{dir_path} prêt")
    
    # Vérifier les fichiers essentiels
    essential_files = [
        "data/SpotifyFeatures.csv",
        "app/streamlit_app_premium.py"
    ]
    
    missing_files = []
    for file_path in essential_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"Fichier manquant: {file_path}")
        else:
            print(f"{file_path} trouvé")
    
    if missing_files:
        print(f"\nFichiers manquants détectés:")
        for file in missing_files:
            print(f"   - {file}")
        
        if "data/SpotifyFeatures.csv" in missing_files:
            print("\n Le fichier de données principal est manquant.")
            print("   L'application utilisera des données de démonstration.")
            input("   Appuyez sur Entrée pour continuer...")
    
    print()
    return True

def create_models():
    """Créer les modèles nécessaires"""
    print("Création des modèles IA...")
    
    # Vérifier si les modèles existent déjà
    model_files = [
        "data/processed/songs_metadata_clean.csv",
        "data/processed/word2vec_similarities.json", 
        "data/processed/content_embeddings.npy"
    ]
    
    models_exist = all(Path(f).exists() for f in model_files)
    
    if models_exist:
        print("Modèles déjà créés")
        recreate = input(" Recréer les modèles? (o/N): ").lower().strip()
        if recreate not in ['o', 'oui', 'y', 'yes']:
            print("⏭  Utilisation des modèles existants")
            print()
            return True
    
    print(" Exécution du script de création des modèles...")
    
    try:
        # Exécuter create_models_manual.py
        result = subprocess.run([sys.executable, "create_models_manual.py"], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(" Modèles créés avec succès!")
            print(result.stdout)
        else:
            print("  Avertissement lors de la création des modèles:")
            print(result.stderr)
            print(" Utilisation du mode démonstration...")
    
    except subprocess.TimeoutExpired:
        print(" Timeout lors de la création - utilisation du mode démo")
    except FileNotFoundError:
        print(" Script create_models_manual.py non trouvé - utilisation du mode démo")
    except Exception as e:
        print(f"  Erreur: {e} - utilisation du mode démo")
    
    print()
    return True

def configure_streamlit():
    """Configurer Streamlit"""
    print(" Configuration de l'interface Streamlit...")
    
    # Créer le fichier de config si nécessaire
    config_path = Path(".streamlit/config.toml")
    if not config_path.exists():
        config_content = """[server]
maxUploadSize = 200
maxMessageSize = 200
headless = true

[theme]
primaryColor = "#1DB954"
backgroundColor = "#121212" 
secondaryBackgroundColor = "#282828"
textColor = "#FFFFFF"
font = "sans serif"

[browser]
gatherUsageStats = false

[global]
developmentMode = false
"""
        
        with open(config_path, "w") as f:
            f.write(config_content)
        print("✅ Configuration Streamlit créée")
    else:
        print("✅ Configuration Streamlit existante")
    
    print()
    return True

def launch_application():
    """Lancer l'application"""
    print(" Lancement de l'application premium...")
    
    app_file = "app/streamlit_app_premium.py"
    
    if not Path(app_file).exists():
        print(f" Application non trouvée: {app_file}")
        
        # Fallback vers la version standard
        fallback_file = "app/streamlit_app_deployment.py"
        if Path(fallback_file).exists():
            print(f" Utilisation de la version standard: {fallback_file}")
            app_file = fallback_file
        else:
            print(" Aucune application trouvée!")
            return False
    
    print(f" Ouverture de {app_file}...")
    print(" L'application va s'ouvrir dans votre navigateur...")
    print()
    print("=" * 60)
    print(" APPLICATION MUSIC RECOMMENDER AI PRÊTE!")
    print("=" * 60)
    print(" Dashboard Analytics - Mode Exploration - Playlist Builder")
    print(" Système de Rating - Feedback - Export CSV")
    print(" Performance optimisée pour déploiement cloud")
    print("=" * 60)
    print()
    print(" CONSEILS POUR LE PORTFOLIO:")
    print("   • Testez toutes les fonctionnalités")
    print("   • Créez quelques playlists démo") 
    print("   • Explorez le dashboard analytics")
    print("   • Notez des recommandations")
    print("   • Capturez des screenshots pour portfolio")
    print()
    print(" Pour arrêter l'application: Ctrl+C")
    print("=" * 60)
    
    try:
        # Lancer Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_file,
            "--server.headless", "true",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n Application arrêtée par l'utilisateur")
    except Exception as e:
        print(f"\n Erreur lors du lancement: {e}")
        return False
    
    return True

def show_deployment_info():
    """Afficher les informations de déploiement"""
    print("\n INFORMATIONS DE DÉPLOIEMENT")
    print("=" * 40)
    
    print(" Pour déployer sur Streamlit Cloud:")
    print("   1. Commitez tous les fichiers sur GitHub")
    print("   2. Allez sur share.streamlit.io") 
    print("   3. Connectez votre repository")
    print("   4. Fichier principal: app/streamlit_app_premium.py")
    print("   5. Branch: main")
    
    print("\n Fichiers créés pour le déploiement:")
    files_created = [
        " app/streamlit_app_premium.py (Application complète)",
        " .streamlit/config.toml (Configuration)",
        " data/processed/ (Modèles et données)",
        " create_models_manual.py (Script de setup)",
        " DEPLOYMENT_GUIDE.md (Guide détaillé)"
    ]
    
    for file_info in files_created:
        print(f"   {file_info}")
    
    print("\n FONCTIONNALITÉS PREMIUM INCLUSES:")
    features = [
        " Dashboard Analytics avec KPIs temps réel",
        " Mode Exploration Intelligent (humeur, époque)",
        " Système de Rating et Feedback utilisateur",
        " Playlist Builder avec export et analyse",
        " Interface premium avec thème Spotify",
        " Visualisations Plotly interactives",
        " Comparaison d'algorithmes avancée",
        " Export CSV et partage social (simulé)"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print("\n Votre application est maintenant prête pour impressionner dans votre portfolio!")

def main():
    """Fonction principale"""
    print_header()
    
    # Étapes de configuration
    steps = [
        (" Vérification des prérequis", check_requirements),
        (" Configuration de l'environnement", setup_environment), 
        (" Création des modèles IA", create_models),
        (" Configuration Streamlit", configure_streamlit)
    ]
    
    # Exécuter chaque étape
    for step_name, step_func in steps:
        print(f" {step_name}")
        try:
            success = step_func()
            if not success:
                print(f" Échec: {step_name}")
                print(" Arrêt du processus de lancement")
                return 1
        except Exception as e:
            print(f" Erreur lors de {step_name}: {e}")
            print(" Tentative de continuation...")
        
        time.sleep(0.5)  # Pause pour la lisibilité
    
    print(" Configuration terminée avec succès!")
    print()
    
    # Demander si l'utilisateur veut lancer maintenant
    launch_now = input(" Lancer l'application maintenant? (O/n): ").lower().strip()
    
    if launch_now in ['', 'o', 'oui', 'y', 'yes']:
        success = launch_application()
        if not success:
            return 1
    else:
        print("  Application configurée mais non lancée")
        print(" Pour lancer manuellement: streamlit run app/streamlit_app_premium.py")
    
    show_deployment_info()
    
    print("\n Mission accomplie! Votre Music Recommender AI est prêt pour le portfolio!")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n Processus interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n Erreur critique: {e}")
        sys.exit(1)