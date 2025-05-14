import joblib
from lightgbm import LGBMClassifier

def train_lightgbm(X_train, y_train, **kwargs):
    model = LGBMClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)