import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
import xgboost as xgb
import catboost as cbo

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import shap

from loguru import logger
from pathlib import Path
import os
import pickle
import yaml
from yaml.loader import SafeLoader
import sys
sys.path.append(os.getcwd())

from src.utils import load_training_data
from src.deployment.classification import Identifier


with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

class xAi:

   @staticmethod
   def add_feature_importance(model, vocabulary, model_folder_path: Path) -> pd.DataFrame:
         ''' 
         This function is used to extract the most important features from a given model. 
         It takes in a trained model and the path of the folder where the model vocabulary is stored. The output of the function is a pandas DataFrame containing the column names, corresponding features, and their importance scores.
         Args:
            model: a trained model object
            vocabulary: vocabulary used to train the model
            model_folder_path: a string representing the path to the folder where the model is stored
         Return: 
            df_features: a pandas DataFrame containing the column names, corresponding features, and their importance scores.
         '''
         path_feature_importance = os.path.join(model_folder_path, "feature_importance.csv")
         if os.path.exists(path_feature_importance):
            df_features = pd.read_csv(path_feature_importance)
         else:
            feature_dict = {vocabulary.shape[0]+index: key for index, key in enumerate(config["dataset_params"]["features_for_model"])}

            # Generate the feature importance values
            if isinstance(model, lgb.LGBMClassifier):
               boost = model.booster_
               importance = boost.feature_importance()
               column = boost.feature_name()
            elif isinstance(model, xgb.Booster):
               importance = model.get_score(importance_type='weight')
               column = list(importance.keys())
            elif isinstance(model, cbo.CatBoostClassifier):
               importance = model.get_feature_importance()
               column = model.feature_names_

            # Store the feature importance
            df_features = pd.DataFrame(columns=['Column','Feature','Importance Score'])
            df_features["Column"] = column
            df_features["Importance Score"] = importance
            for j in range(len(column)):
               if j < vocabulary.shape[0]:
                     df_features.loc[j,"Feature"] = vocabulary[j]
               else:
                     df_features.loc[j,"Feature"] = feature_dict[j]

            df_features = df_features.sort_values(by=['Importance Score'], ascending=False)

            df_features.to_csv(path_feature_importance)

         return df_features

   @staticmethod
   def get_features(df_features: pd.DataFrame) -> tuple[list, list]:
      ''' 
      This function is used to retrieve the list of features and their importance scores from a given path of a folder where the features are stored in an Excel file. 
      The function returns two lists - a complete list of feature names and a list of top 20 most important features.
      Args:
         model_folder_path: a string representing the path of the folder where the features are stored
      Return:
         feature_list: a list of strings representing all features stored in the Excel file
         topx_important_features: a list of integers representing the indices of the 20 most important features in the feature_list. If the Excel file does not exist, a message indicating the error is printed and the function returns None.
      '''

      # Get top 20 features
      topx_important_features = df_features.head(20)
      topx_important_features = topx_important_features.index.tolist()
      feature_list = df_features["Feature"].values.tolist()

      return feature_list, topx_important_features

   @staticmethod
   def plot_shap_summary(model, vocabulary, dataset_folder, model_folder_path: Path):

      df_features = xAi.add_feature_importance(model, vocabulary, model_folder_path)
      feature_list, topx_important_features = xAi.get_features(df_features)

      # Create the explainer object
      explainer = shap.TreeExplainer(model)

      # Load dataset
      X_train, y_train, X_val, y_val, X_test, y_test, df_preprocessed, df_train, df_val, df_test, weight_factor = load_training_data(dataset_folder, binary_model=True)

      X = np.concatenate((X_train, X_val), axis=0)
      y = np.concatenate((y_train, y_val), axis=0)

      try:
         # Get shap values for all features
         shap_values = explainer.shap_values(X, y)
         plt.clf()
         shap.summary_plot(shap_values[1], X, feature_list, show=False)
         plt.savefig(os.path.join(model_folder_path, "shap_top10_features.png"))
      except:
          logger.info("Shap plot can not be generated.")

      return df_features, X, y

   @staticmethod
   def create_tree(model, X, y, model_path: Path, tree_index: int):
      os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"
      plt.clf()
      if isinstance(model, lgb.Booster):
         ax = lgb.plot_tree(model.booster_, orientation='vertical', tree_index=tree_index, figsize=(20, 8), show_info=['split_gain'])
      elif isinstance(model, xgb.Booster):
         ax = xgb.plot_tree(model, num_trees=tree_index)
      elif isinstance(model, cbo.CatBoostClassifier):
         pool = cbo.Pool(X, y, feature_names=list(X.columns))
         ax = cbo.plot_tree(model.booster_, orientation='vertical', tree_idx=tree_index, figsize=(20, 8), show_info=['split_gain'])

      plt.savefig(os.path.join(model_path, "xAi_tree.png"))

   @staticmethod
   def create_path(path: Path):
      isExist = os.path.exists(path)
      if not isExist:
         os.makedirs(path)
      
def main():
   logger.info("Store plots to explain the binary model...")
   model_path_binary = Path("final_models/Binary_model")
   model, vectorizer, vocabulary, bbox_features = Identifier.get_model(model_path_binary)
   xai_folder_path = os.path.join(model_path_binary, "xAi")
   xAi.create_path(xai_folder_path)
   dataset_folder = Identifier.search_in_logging(text="Dataset:", model_folder_path=model_path_binary)
   df_features, X, y = xAi.plot_shap_summary(model, vocabulary,dataset_folder, model_folder_path=xai_folder_path)
   xAi.create_tree(model, X, y, xai_folder_path, tree_index=300)

   logger.info("Store plots to explain the multiclass model...")
   model_path_multiclass = "final_models/Multiclass_model/"
   model, vectorizer, vocabulary, bbox_features = Identifier.get_model(model_path_multiclass)
   xai_folder_path = os.path.join(model_path_multiclass, "xAi")
   xAi.create_path(xai_folder_path)
   dataset_folder = Identifier.search_in_logging(text="Dataset:", model_folder_path=model_path_binary)
   df_features, X, y = xAi.plot_shap_summary(model, vocabulary, dataset_folder, model_folder_path=xai_folder_path)
   xAi.create_tree(model, X, y, xai_folder_path, tree_index=300)

   logger.info("xAi plots successfully stored!")

if __name__ == "__main__":
    
    main()