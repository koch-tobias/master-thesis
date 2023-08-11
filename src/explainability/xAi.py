import numpy as np
import matplotlib.pyplot as plt

import shap
import lightgbm as lgb

from src.deployment_pipeline.prediction import get_model
from src.training_pipeline.utils import load_dataset

import yaml
from yaml.loader import SafeLoader
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

def plot_shap_summary(model):

   # Create the explainer object
   explainer = shap.TreeExplainer(model)

   #Load dataset
   X_train, y_train, X_val, y_val, X_test, y_test, df_preprocessed, df_test, weight_factor = load_dataset(binary_model=True)

   # Get shap values for all features
   shap_values = explainer(X_val)

   # Calculate the mean absolute Shapley values for each feature
   mean_abs_shap = np.abs(shap_values).mean(axis=0)

   # Get the indices of the top 10 features by mean absolute Shapley value
   top_features = mean_abs_shap.argsort()[-10:]

   # Create the summary plot for the top features only
   shap.summary_plot(shap_values[:, top_features], X_val[:, top_features])

   plt.savefig(config["paths"]["modelpath"] + "/shap_top10_features.png")

def main():
   model, vectorizer, vocabulary, bbox_features = get_model(config["paths"]["modelpath"])
   plot_shap_summary(model)

if __name__ == "__main__":
    
    main()