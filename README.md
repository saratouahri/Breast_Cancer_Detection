# Breast Cancer Detection with Machine Learning

Projet de classification binaire pour prédire si une tumeur du sein est bénigne ou maligne.

## Structure du projet
- **data/**: dataset au format CSV
- **notebooks/**: notebooks d'exploration et d'entraînement
- **src/**: modules Python réutilisables (prétraitement, entraînement, évaluation, explication)
- **app/**: application Streamlit interactive

## Installation
```bash
git clone https://github.com/ton-compte/breast_cancer_detection_project.git
cd breast_cancer_detection_project
pip install -r requirements.txt
```

## Usage
- Lancer les notebooks pour explorer et entraîner
- `python src/train_model.py` pour entraîner et sauvegarder le modèle
- `streamlit run app/app.py` pour démarrer l'application web

## Résultats clés
- Accuracy: ~97%
- Recall (cancer): ~99%

## Déploiement
- Hébergez sur Streamlit Cloud ou Hugging Face Spaces