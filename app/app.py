import os, sys

# ─── Ajouter la racine du projet ────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ─── Imports ───────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

from src.preprocess  import load_data, preprocess_data
from src.train_model import load_model, train_lightgbm

# ─── Chargement des données + prétraitement (TOUJOURS exécuté) ────────────────
df = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# ─── Chargement ou (re)entraînement du modèle ─────────────────────────────────
MODEL_PATH = os.path.join(project_root, 'models', 'lightgbm_model.pkl')
if os.path.exists(MODEL_PATH):
    model, _ = load_model(MODEL_PATH)
else:
    model = train_lightgbm(X_train, y_train)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump((model, scaler), MODEL_PATH)

# ─── Titre et description ─────────────────────────────────────────────────────
st.title("Breast Cancer Detection with ML + SHAP")
st.markdown("Entrez les caractéristiques de la tumeur pour obtenir la prédiction et l'explication SHAP.")

# ─── Sliders pour chaque feature ────────────────────────────────────────────────
features = df.drop('target', axis=1).columns.tolist()
user_input = {}
for feat in features:
    mi, ma, me = float(df[feat].min()), float(df[feat].max()), float(df[feat].mean())
    user_input[feat] = st.slider(feat, mi, ma, me)

# ─── Bouton de prédiction ─────────────────────────────────────────────────────
if st.button("Prédire"):
    # 1) Prépare l'entrée
    X_df        = pd.DataFrame([user_input])
    X_df_scaled = pd.DataFrame(scaler.transform(X_df), columns=X_df.columns)

    # 2) Prédiction
    pred = model.predict(X_df_scaled)
    st.write("🔬 Prédiction :", "**Malin**" if pred[0] == 1 else "**Bénin**")

    # 3) Explication SHAP (waterfall pour un seul cas)
    explainer   = shap.Explainer(model, X_train)      # X_train en “background”
    shap_values = explainer(X_df_scaled)

    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
