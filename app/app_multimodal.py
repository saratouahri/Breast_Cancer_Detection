# app_multimodal.py – Multimodal model with tabular SHAP (via shap.Explainer) et gradient saliency map
import os
import sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
import shap
import joblib
from torchvision import transforms
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from src.multimodal_model import MultimodalNet
# —————————————————————————————————————————————
# 0) User guide in the sidebar
st.sidebar.title("💡 How to interpret the results")
st.sidebar.markdown("""
**1. Prediction**  
- **Benign / Malignant**: the verdict from the multimodal model  
- **Probabilities**: confidence score for each class  

**2. Tabular feature importance (SHAP)**  
- *Waterfall plot*:  
  - **Base value** = average model output (E[f(X)])  
  - Each bar shows how a feature pushes the prediction up (red, +) or down (blue, –)  
  - Sum of contributions + base value = final probability  

**3. Image saliency map**  
- Overlaid heatmap on the biopsy image   
- **Gray / dark red**: little to no influence  
- **Red**: moderate influence  
- **Orange / yellow**: strongest influence  
""")
# 1) Données tabulaires + normalisation
data = load_breast_cancer()
X_tab = data['data'].astype(np.float32)
feature_names = data['feature_names']
scaler = StandardScaler().fit(X_tab)
X_tab = scaler.transform(X_tab)

# 2) Modèle multimodal
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalNet(tabular_input_dim=X_tab.shape[1])
model.load_state_dict(torch.load(
    os.path.join(project_root, 'models', 'multimodal_model.pt'),
    map_location=device
))
for m in model.modules():
    if isinstance(m, torch.nn.ReLU):
        m.inplace = False
model.to(device).eval()

# 3) Chargement du LightGBM tabulaire
lgbm_raw = joblib.load(os.path.join(project_root, 'models', 'lightgbm_model.pkl'))
if isinstance(lgbm_raw, tuple):
    for candidate in lgbm_raw:
        if hasattr(candidate, "predict_proba") or hasattr(candidate, "predict"):
            lgbm = candidate
            break
else:
    lgbm = lgbm_raw

# 4) Interface Streamlit
st.title("Breast Cancer Detection (Multimodal)")
st.write("Enter tumor features and upload a biopsy image for prediction and explanation.")

# 5) Sliders pour features tabulaires
user_input = {}
for i, name in enumerate(feature_names):
    col = X_tab[:, i]
    user_input[name] = st.slider(name, float(col.min()), float(col.max()), float(col.mean()))

X_user = pd.DataFrame([user_input], dtype=np.float32)
X_scaled = scaler.transform(X_user)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

# 6) Uploader image
img_file = st.file_uploader("Upload a biopsy image", type=['png','jpg','jpeg'])
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3,1,1))
])

if img_file:
    # Préparation image
    image = Image.open(img_file).convert('L')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 7) Prédiction multimodale
    with torch.no_grad():
        output = model(image_tensor, X_tensor)
        prob_malign = float(output.item())
        prob_benign = 1.0 - prob_malign
        label = "Malignant" if prob_malign > 0.5 else "Benign"
    st.write(f"Prediction: **{label}**")
    st.write(f"Probability Benign: **{prob_benign:.1%}**")
    st.write(f"Probability Malignant: **{prob_malign:.1%}**")

    # 8) Explication tabulaire via shap.Explainer
    st.subheader("Tabular feature importance (SHAP – LightGBM)")
    if hasattr(lgbm, "predict_proba"):
        f = lambda x: lgbm.predict_proba(x)[:, 1]
    else:
        f = lambda x: lgbm.predict(x)

    explainer = shap.Explainer(f, X_tab, feature_names=feature_names)
    shp = explainer(X_scaled)

    vals     = shp.values[0]
    base     = shp.base_values[0]
    data_row = shp.data[0]

    exp = shap.Explanation(
        values       = vals,
        base_values  = base,
        data         = data_row,
        feature_names= feature_names
    )

    # Création explicite de la figure avant appel
    plt.figure(figsize=(8,5))
    shap.waterfall_plot(exp, max_display=10, show=False)
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # 9) Saliency map sur l’image
    st.subheader("Image pixel importance (saliency map)")
    image_tensor.requires_grad_()
    model.zero_grad()
    pred = model(image_tensor, X_tensor)
    pred.backward()
    saliency = image_tensor.grad.abs().squeeze().cpu().numpy().transpose(1,2,0)

    plt.figure(figsize=(6,6))
    plt.imshow(image.resize((224,224)), cmap='gray', alpha=0.6)
    plt.imshow(np.sum(saliency, axis=2), cmap='hot', alpha=0.4)
    plt.axis('off')
    st.pyplot(plt.gcf())

else:
    st.info("Please upload an image to proceed.")



# —————————————————————————————————————————————
import streamlit as st
# … your existing imports and model loading …





