# Breast Cancer Detection with Machine Learning

A Streamlit app that combines **tabular data** (LightGBM + SHAP explanations) and **biopsy images** (ResNet18 + saliency maps) to predict whether a breast tumor is benign or malignant .

---
## Project Structure
- **data/**: datasets in CSV format  
- **notebooks/**: exploratory analysis and training notebooks  
- **src/**: reusable Python modules (preprocessing, training, evaluation, explanation)  
- **app/**: interactive Streamlit application  

## Installation

```bash
git clone https://github.com/saratouahri/Breast_Cancer_Detection.git
cd Breast_Cancer_Detection.git
pip install -r requirements.txt
```
## Usage
Train the model

```
python src/train_model.py \
  --input data/breast_cancer.csv \
  --output models/lightgbm_model.pkl

```
Run the Streamlit app
```
streamlit run app/app.py


```
## Key Results
**Confusion Matrix**
![Confusion matrix of the model](app/cm.png)


**Metrics**
Accuracy: 0.7273

AUC-ROC: 0.7466
