import warnings 
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger

import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig, TabNetModelConfig, FTTransformerConfig 
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer

import sys
sys.path.append('C:/Users/q617269/Desktop/Masterarbeit_Tobias/master-thesis')

from src.training_pipeline.utils import load_dataset

import yaml
from yaml.loader import SafeLoader
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

def main():
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, df_preprocessed, df_train, df_val, df_test, weight_factor = load_dataset(binary_model=True)

    relevant_columns = config["general_params"]["features_for_model"] + ["Benennung (bereinigt)", "Relevant fuer Messung"]
    df_train = df_train[relevant_columns]
    df_val = df_val[relevant_columns]
    df_test = df_test[relevant_columns]

    df_train['Relevant fuer Messung'] = df_train['Relevant fuer Messung'].map({'Ja':1, 'Nein':0}).astype("category")
    df_val['Relevant fuer Messung'] = df_val['Relevant fuer Messung'].map({'Ja':1, 'Nein':0}).astype("category")
    df_test['Relevant fuer Messung'] = df_test['Relevant fuer Messung'].map({'Ja':1, 'Nein':0}).astype("category")

    df_train["Benennung (bereinigt)"] = df_train["Benennung (bereinigt)"].astype("category").cat.codes
    df_val["Benennung (bereinigt)"] = df_val["Benennung (bereinigt)"].astype("category").cat.codes
    df_test["Benennung (bereinigt)"] = df_test["Benennung (bereinigt)"].astype("category").cat.codes

    # Setting up the data configs
    data_config = DataConfig(
                                target=["Relevant fuer Messung"], #target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
                                continuous_cols=config["general_params"]["features_for_model"],
                                categorical_cols=["Benennung (bereinigt)"],
                                continuous_feature_transform="quantile_normal",
                                normalize_continuous_features=True
                            )

    # Setting up trainer configs
    trainer_config = TrainerConfig(
                                    auto_lr_find=True, # Runs the LRFinder to automatically derive a learning rate
                                    batch_size=32,
                                    max_epochs=100,
                                    gpus=1, #index of the GPU to use. 0, means CPU
                                )

    # Setting up optimizer configs
    optimizer_config = OptimizerConfig()

    # Setting up model configs
    model_config = CategoryEmbeddingModelConfig(
                                                    task="classification",
                                                    layers="128-64-32",  # Number of nodes in each layer
                                                    activation="LeakyReLU", # Activation between each layers
                                                    learning_rate = 1e-3,
                                                    metrics=["f1_score"]
                                                )

    # Initialize model
    tabular_model = TabularModel(
                                    data_config=data_config,
                                    model_config=model_config,
                                    optimizer_config=optimizer_config,
                                    trainer_config=trainer_config
                                )

    tabular_model.fit(train=df_train, test=df_val)

    tabular_model.evaluate(df_val)
    result = tabular_model.evaluate(df_test)
    pred_df = tabular_model.predict(df_test)
    wrong_predictions = pd.DataFrame(columns=pred_df.columns)
    for index, row in pred_df.iterrows():
        if row["Relevant fuer Messung"] != row["prediction"]:
            wrong_predictions = pd.concat([wrong_predictions, pd.DataFrame([row])], ignore_index=True)

    tabular_model.summary()
    wrong_predictions.to_excel('saved_models/wrong_predictions.xlsx')
    pred_df.to_excel('saved_models/pred_df.xlsx')
    tabular_model.save_model('saved_models/binary_model')

# %%
if __name__ == "__main__":
    
    main()

