import os
import sys

# ─── Ajouter la racine du projet au PYTHONPATH ────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ─── Imports ───────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.preprocess    import load_data, preprocess_data
from src.train_model   import load_model, train_lightgbm
from src.evaluate      import evaluate_model, plot_confusion_matrix

# ─── Chargement ou entraînement du modèle ────────────────────────────────────
MODEL_PATH = os.path.join(project_root, 'models', 'lightgbm_model.pkl')
if os.path.exists(MODEL_PATH):
    model, scaler = load_model(MODEL_PATH)
else:
    # (Re)entraîner si pas de modèle enregistré
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    model = train_lightgbm(X_train, y_train)
    # Sauvegarder pour la prochaine fois
    import joblib
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump((model, scaler), MODEL_PATH)

# ─── Titre de l’application ───────────────────────────────────────────────────
st.title("Breast Cancer Detection with ML + SHAP")
st.markdown("Entrez les caractéristiques de la tumeur pour obtenir la prédiction et l'explication SHAP.")

# ─── Chargement des données pour définir les sliders ─────────────────────────
df = load_data()
features = df.drop('target', axis=1).columns.tolist()

# ─── Création des sliders utilisateur ────────────────────────────────────────
user_input = {}
for feat in features:
    min_val  = float(df[feat].min())
    max_val  = float(df[feat].max())
    mean_val = float(df[feat].mean())
    user_input[feat] = st.slider(feat, min_val, max_val, mean_val)

# ─── Bouton de prédiction ─────────────────────────────────────────────────────
if st.button("Prédire"):
    X_df     = pd.DataFrame([user_input])
    X_scaled = scaler.transform(X_df)

    # Prédiction
    pred = model.predict(X_scaled)
    st.write("🔬 Prédiction :", "**Malin**" if pred[0] == 1 else "**Bénin**")

    # ─── Explication SHAP ──────────────────────────────────────────────────────
    # On utilise X_train comme "background dataset" si disponible, sinon X_scaled
    background = locals().get('X_train', None)
    explainer  = shap.Explainer(model, background if background is not None else X_scaled)
    shap_values = explainer(X_scaled)

    # 1) Summary plot (influence globale)
    fig_summary = plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_scaled, show=False)
    st.pyplot(fig_summary)

    # 2) Waterfall plot (explication individuelle)
    fig_water = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig_water)
