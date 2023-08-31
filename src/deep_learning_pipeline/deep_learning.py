import warnings 
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

import tensorflow as tf
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig, TabNetModelConfig, FTTransformerConfig 
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer

from loguru import logger
import sys
import time
import yaml
import json
import os
import yaml
from yaml.loader import SafeLoader
sys.path.append(os.getcwd())

from src.utils import load_dataset


with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

def get_metrics_results(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    val_acc = accuracy_score(y_true, y_pred)
    val_sensitivity = recall_score(y_true, y_pred, average='macro')
    val_f1 = f1_score(y_true, y_pred, average='macro')

    return val_acc, val_sensitivity, val_f1

def model_architecture(lr, dropout, batch_size, epochs, patience, activation, layers):
    # Setting up the data configs
    data_config = DataConfig(
                                target=["Relevant fuer Messung"], #target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
                                continuous_cols=config["dataset_params"]["features_for_model"],
                                categorical_cols=["Benennung (bereinigt)"],
                                continuous_feature_transform="quantile_normal",
                                normalize_continuous_features=True
                            )

    # Setting up trainer configs
    trainer_config = TrainerConfig(
                                    auto_lr_find=True, # Runs the LRFinder to automatically derive a learning rate
                                    batch_size=batch_size,
                                    max_epochs=epochs,
                                    early_stopping_patience=patience,
                                    gpus=1, #index of the GPU to use. 0, means CPU
                                )

    # Setting up optimizer configs
    optimizer_config = OptimizerConfig()

    # Setting up model configs
    model_config = CategoryEmbeddingModelConfig(
                                                    task="classification",
                                                    layers=layers,  # Number of nodes in each layer
                                                    activation=activation, # Activation between each layers
                                                    learning_rate = lr,
                                                    initialization = "kaiming",
                                                    dropout=dropout,
                                                    metrics=["accuracy", "f1_score"],
                                                    metrics_params=[{},{"average":"macro"}],
                                                    metrics_prob_input=[False, True]
                                                )

    experiment_config = ExperimentConfig(project_name="experiments", 
                                        exp_watch="all", 
                                        log_target="tensorboard")
    # Initialize model
    tabular_model = TabularModel(
                                    data_config=data_config,
                                    model_config=model_config,
                                    optimizer_config=optimizer_config,
                                    trainer_config=trainer_config,
                                    experiment_config=experiment_config
                                )

    return tabular_model

def create_logfile(model_folder_path, val_auc, best_iteration, index_best_model, metrics, hp):
    logging_file_path = model_folder_path + "logging.txt"
    if os.path.isfile(logging_file_path):
        log_text = "Validation AUC (final model): {}\n".format(val_auc)
        f= open(model_folder_path + "logging.txt","a")
        f.write("\n_________________________________________________\n")
        f.write("Final model:\n")
        f.write(log_text)
        f.write("Trained Iterations: {}\n".format(best_iteration))
        f.close()
    else:
        dataset_path = "Dataset: {}\n".format(config["paths"]["folder_processed_dataset"])
        model_folder = "Model folder path: {}\n".format(model_folder_path)
        f= open(model_folder_path + "logging.txt","w+")
        f.write(dataset_path)
        f.write(model_folder)
        f.write("use_only_text: {}".format(config["dataset_params"]["use_only_text"]))
        f.write("\n_________________________________________________\n")
        f.write("Best model after hyperparameter tuning:\n")
        f.write("Validation AUC: {}\n".format(val_auc))
        f.write("Trained Iterations: {}\n".format(best_iteration))
        f.write("Model index in hyperparameter tuning: {}\n".format(index_best_model+1))
        f.write("Hyperparameter:\n")
        for key in hp:
            f.write("{}: {}\n".format(key, hp[key]))
        f.write(f"Metrics: {metrics} \n")
        f.write(json.dumps(config["train_settings"]))
        f.write("\n")
        f.write(json.dumps(config["prediction_settings"]))
        f.write("\n")
        f.close()   

def results_from_experiment(experiment_path, metric):
    results_list = []
    for e in tf.compat.v1.train.summary_iterator(experiment_path):
        for v in e.summary.value:
            if v.tag == metric:
               results_list.append(v.simple_value)

def get_model_results(experiment_path):

    result_dict = { 
                    "train_loss": [],
                    "train_accuracy": [], 
                    "train_f1_score": [],
                    "val_loss": [],
                    "val_accuracy": [],
                    "val_f1_score": []
                }
    
    for metric in result_dict.keys():
        result_dict[metric] = results_from_experiment(experiment_path, metric)

    return result_dict

def main():

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, df_preprocessed, df_train, df_val, df_test, weight_factor = load_dataset(binary_model=True)

    # Keep only relevant features
    relevant_columns = config["dataset_params"]["features_for_model"] + ["Benennung (bereinigt)", "Relevant fuer Messung"]
    df_train = df_train[relevant_columns]
    df_val = df_val[relevant_columns]
    df_test = df_test[relevant_columns]

    # Map relevant car parts to 1 and not relevant car parts to 0
    df_train['Relevant fuer Messung'] = df_train['Relevant fuer Messung'].map({'Ja':1, 'Nein':0}).astype("category")
    df_val['Relevant fuer Messung'] = df_val['Relevant fuer Messung'].map({'Ja':1, 'Nein':0}).astype("category")
    df_test['Relevant fuer Messung'] = df_test['Relevant fuer Messung'].map({'Ja':1, 'Nein':0}).astype("category")

    # Convert car part designation to category
    df_train["Benennung (bereinigt)"] = df_train["Benennung (bereinigt)"].astype("category").cat.codes
    df_val["Benennung (bereinigt)"] = df_val["Benennung (bereinigt)"].astype("category").cat.codes
    df_test["Benennung (bereinigt)"] = df_test["Benennung (bereinigt)"].astype("category").cat.codes

    # Set hyperparameter
    hp_dict = {
        'layers': ["32-16-8", "64-32-16", "128-64-32"],
        'activation_functions': ["LeakyReLU", "ReLU", "sigmoid"],
        'batch_size': [64, 32, 16],
        'dropout': [0, 0.01, 0.1]
    }

    # Declare the column names to generate the dataframe to store the results and the hyperparameter of the models using grid search
    fix_columns = ["Model nr", "train_accuracy", "Train f1_score", "Train loss", "Validation acc", "Validation f1_score", "Validation loss", "Test acc", "Test sensitivity", "Test f1_score","Training time (s)"]
    hp_columns = list(hp_dict.keys())
    columns = fix_columns + hp_columns

    # Create empty dataframe with the predefined column names
    df_gridsearch = pd.DataFrame(columns=columns)

    # Declare list to store the trained models
    trained_models = {}

    # Grid search hyper parametertuning
    model_nr = 1
    for hp_0 in hp_dict[0]:
        for hp_1 in hp_dict[1]:
            for hp_2 in hp_dict[2]:
                for hp_3 in hp_dict[3]:   
                    # Build the model archtecture
                    tabular_model = model_architecture(lr= 1e-3, dropout=hp_3, batch_size=hp_2, epochs=100, patience=10, activation=hp_1, layers=hp_0)
                    
                    # Train and evaluate the model
                    start = time.time()
                    tabular_model.fit(train=df_train, validation=df_val)
                    stop = time.time()
                    training_time = int(stop - start)
                    tabular_model.evaluate(df_val)
                    
                    # Get results after training
                    result_dict = get_model_results(experiment_path)

                    # Predictions on testset
                    pred_df = tabular_model.predict(df_test)
                    test_acc, test_sensitivity, test_f1 = get_metrics_results(y_true=pred_df["Relevant fuer Messung"], y_pred=pred_df["prediction"])

                    # Add model and results to list of trained models
                    str_model_nr = "model_nr " + str(model_nr)
                    trained_models[str_model_nr] = []
                    trained_models[str_model_nr].append(tabular_model)
                    trained_models[str_model_nr].append(result_dict)

                    # Get index of best iteration 


                    # Store model results in a dataframe
                    df_gridsearch.loc[model_nr, "Model nr"] = model_nr
                    df_gridsearch.loc[model_nr, "Train loss"] = result_dict["train_loss"]
                    df_gridsearch.loc[model_nr, "Train acc"] = result_dict["train_accuracy"]
                    df_gridsearch.loc[model_nr, "Train f1_score"] = result_dict["train_f1_score"]
                    df_gridsearch.loc[model_nr, "Validation loss"] = result_dict["validation_loss"]
                    df_gridsearch.loc[model_nr, "Validation acc"] = result_dict["validation_accuracy"]
                    df_gridsearch.loc[model_nr, "Validation f1_score"] = result_dict["validation_f1_score"]
                    df_gridsearch.loc[model_nr, "Test acc"] = test_acc
                    df_gridsearch.loc[model_nr, "Test sensitivity"] = test_sensitivity
                    df_gridsearch.loc[model_nr, "Test f1_score"] = test_f1                  
                    df_gridsearch.loc[model_nr, "Training time (s)"] = training_time
                    df_gridsearch.loc[model_nr, hp_columns[0]] = hp_0
                    df_gridsearch.loc[model_nr, hp_columns[1]] = hp_1
                    df_gridsearch.loc[model_nr, hp_columns[2]] = hp_2
                    df_gridsearch.loc[model_nr, hp_columns[3]] = hp_3

                    model_nr = model_nr + 1
                    
                    break
                break
            break
        break
    '''
    df_gridsearch.to_csv("results_grid_search.csv")

    # Predictions on the testset
    pred_df = tabular_model.predict(df_test)

    # Store wrong predictions on the testset
    wrong_predictions = pd.DataFrame(columns=pred_df.columns)
    for index, row in pred_df.iterrows():
        if row["Relevant fuer Messung"] != row["prediction"]:
            wrong_predictions = pd.concat([wrong_predictions, pd.DataFrame([row])], ignore_index=True)
    wrong_predictions.to_excel('saved_models/wrong_predictions.xlsx')

    # Store the best model after hyperparameter tuning and crossvalidation
    tabular_model.save_model('saved_models/binary_model')

    # Store the results to logging file

    # Store the model summary to pkl file
    tabular_model.summary()
    '''

# %%
if __name__ == "__main__":
    
    main()

