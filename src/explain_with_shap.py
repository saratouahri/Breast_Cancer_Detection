import shap

def explain_with_shap(model, X, sample_index=0):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    # résumé global
    shap.summary_plot(shap_values, X)
    # explication individuelle
    shap.plots.waterfall(shap_values[sample_index])