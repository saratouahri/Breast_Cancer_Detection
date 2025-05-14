
import sys, os

# Racine de ton projet (un cran au-dessus de ce fichier)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Insère la racine du projet en tête de sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import streamlit as st
import pandas as pd
from src.preprocess import load_data, preprocess_data
from src.train_model import load_model
from src.explain_with_shap import explain_with_shap

# Charger modèle et données
model, scaler = load_model('models/lightgbm_model.pkl')
df = load_data()

st.title('Breast Cancer Detection')
st.markdown('Entrez les caractéristiques pour prédiction:')
features = df.drop('target', axis=1).columns.tolist()
user_input = {}
for feat in features:
    min_val, max_val = float(df[feat].min()), float(df[feat].max())
    user_input[feat] = st.slider(feat, min_val, max_val, float(df[feat].mean()))
if st.button('Prédire'):
    X_df = pd.DataFrame([user_input])
    X_scaled = scaler.transform(X_df)
    pred = model.predict(X_scaled)
    st.write('Prédiction :', 'Malin' if pred[0]==1 else 'Bénin')
    explain_with_shap(model, X_df)