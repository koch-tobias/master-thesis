import warnings 
warnings.filterwarnings('ignore')

import pandas as pd
import math
from statistics import mean
from sklearn.metrics import accuracy_score, f1_score, recall_score

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import StratifiedKFold


import tensorflow as tf
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig, TabNetModelConfig, FTTransformerConfig 
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer

from loguru import logger
from pathlib import Path
import sys
import time
import yaml
import json
import os
import yaml
from datetime import datetime
from yaml.loader import SafeLoader
sys.path.append(os.getcwd())

from src.utils import load_training_data

with open('src/deep_learning_pipeline/config_dl.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)


def get_metrics_results(y_true, y_pred):
    result_dict = {}
    
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    result_dict["test_accuracy"] = accuracy_score(y_true, y_pred)
    result_dict["test_sensitivity"] = recall_score(y_true, y_pred, average='macro')
    result_dict["test_f1_score"] = f1_score(y_true, y_pred, average='macro')

    return result_dict

def model_architecture(lr, dropout, batch_size, activation, layers, experiment_path, run_name):
    # Setting up the data configs
    data_config = DataConfig(
                                target=config["target"],
                                continuous_cols=config["continuous_cols"],
                                categorical_cols=config["categorical_cols"],
                                continuous_feature_transform=config["continuous_feature_transform"],
                                normalize_continuous_features=config["normalize_continuous_features"]
                            )

    # Setting up trainer configs
    trainer_config = TrainerConfig(
                                    auto_lr_find=True,                                                # Runs the LRFinder to automatically derive a learning rate
                                    batch_size=batch_size,
                                    max_epochs=config["max_epochs"],
                                    early_stopping=config["earlystopping_metric"],
                                    early_stopping_min_delta=config["early_stopping_min_delta"],
                                    early_stopping_mode=config["early_stopping_mode"], 
                                    early_stopping_patience=config["early_stopping_patience"], 
                                    checkpoints=config["earlystopping_metric"], 
                                    checkpoints_path=experiment_path,
                                    load_best=True,                                                   # After training, load the best checkpoint
                                )

    # Setting up optimizer configs
    optimizer_config = OptimizerConfig()

    # Setting up model configs
    model_config = CategoryEmbeddingModelConfig(
                                                    task=config["task"],
                                                    layers=layers,  
                                                    activation=activation,
                                                    learning_rate = lr,
                                                    dropout=dropout,
                                                    initialization = config["weight_init"],
                                                    loss= config["loss"],
                                                    metrics=["accuracy", "f1_score"],
                                                    metrics_params=[{},{"average":"macro"}],
                                                    metrics_prob_input=[True, True]
                                                )

    experiment_config = ExperimentConfig(project_name=experiment_path, 
                                        run_name=run_name,
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


def create_logfile(model_folder_path, data_folder, train_loss, train_accuracy, train_f1_score, valid_loss, valid_accuracy, valid_f1_score, dropout, layers, activation, batch):
    logging_file_path = os.path.join(model_folder_path, "logging.txt")

    dataset_path = "Dataset: {}\n".format(data_folder)
    model_folder = "Model folder path: {}\n".format(model_folder_path)
    f = open(logging_file_path,"w+")
    f.write(dataset_path)
    f.write(model_folder)
    f.write("\n_________________________________________________\n")
    f.write("Results:\n")
    f.write("Training loss: {}\n".format(train_loss))
    f.write("Training accuracy: {}\n".format(train_accuracy))
    f.write("Training F1-Score: {}\n".format(train_f1_score))
    f.write("Validation loss: {}\n".format(valid_loss))
    f.write("Validation accuracy: {}\n".format(valid_accuracy))
    f.write("Validation F1-Score: {}\n".format(valid_f1_score))
    f.write("\n_________________________________________________\n")
    f.write("Hyperparameter:\n")
    f.write("Layer: {}\n".format(layers))
    f.write("Dropout: {}\n".format(dropout))
    f.write("Activation function: {}\n".format(activation))
    f.write("Batch Size: {}\n".format(batch))
    f.close()   

def results_from_experiment(model_path, metric):
    results_list = []
    for e in tf.compat.v1.train.summary_iterator(model_path):
        for v in e.summary.value:
            if v.tag == metric:
                results_list.append(v.simple_value)
    
    return results_list

def get_model_results(model_path):

    result_dict = { 
                    "train_loss": [],
                    "train_accuracy": [], 
                    "train_f1_score": [],
                    "valid_loss": [],
                    "valid_accuracy": [],
                    "valid_f1_score": [],
                    "epoch": []
                }
    
    for metric in result_dict.keys():
        result_dict[metric].extend(results_from_experiment(model_path, metric))

    return result_dict

def find_event_file(folder_path):
   for file in os.listdir(folder_path):
       if file.startswith("events"):
           return os.path.join(folder_path, file)
   return None

def store_metrics(best_iteration, train_results, validation_results, metric, model_folder_path):
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.plot(train_results, label=f'Train {metric}')
    plt.plot(validation_results, label=f'Validation {metric}')
    plt.xlabel('Number of Iterations')
    plt.ylabel(f'{metric}')
    plt.title(f'Training and Validation {metric}')
    plt.axvline(best_iteration, color='b', label = 'Early Stopping')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(os.path.join(model_folder_path, f'{metric}_plot.png'))
    plt.close()

def store_confusion_matrix(df_test: pd.DataFrame, df_pred: pd.DataFrame, model_folder_path: Path):
    ''' 
    This function stores the confusion matrix of a machine learning model, given the test labels and predicted labels.
    It saves the plot in a specified folder. The function also checks whether the model is binary or multiclass, in order to design the plot and the class names. If multiclass, it loads a label encoder from a saved file. 
    Args:
        y_test: numpy array containing the true labels of the test set.
        y_pred: numpy array containing the predicted labels of the test set.
        folder_path: string containing the path where the label_encoder.pkl file is stored.
        model_folder_path: string containing the path where the resulting plot should be saved.
        binary_model: boolean variable indicating whether the model is binary or not. 
    Return: None 
    '''
    class_names = ["Not relevant", "Relevant"]
    plt.rcParams["figure.figsize"] = (15, 15)
    ConfusionMatrixDisplay.from_predictions(df_test, df_pred, display_labels=class_names, cmap='Blues', colorbar=False,  text_kw={'fontsize': 12})
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12 )
    plt.ylabel('True Label', fontsize=12)
    plt.savefig(os.path.join(model_folder_path, 'confusion_matrix.png'))

    plt.close('all')

def grid_search(df_train, df_val, df_test, experiment_path, timestamp):
    # Set hyperparameter
    hp_dict = config["hyperparameter"]
    # Declare the column names to generate the dataframe which stores the results and the hyperparameter of the models using grid search
    fix_columns = ["Model", "Train loss",  "Train accuracy", "Train f1-score", "Validation loss", "Validation accuracy", "Validation f1-score", "Test accuracy", "Test sensitivity", "Test f1-score", "Training time (s)", "Trained epochs"]
    hp_columns = list(hp_dict.keys())
    columns = fix_columns + hp_columns

    # Create empty dataframe with the predefined column names
    df_gridsearch = pd.DataFrame(columns=columns)

    # Declare a dict to store the trained models
    trained_models = {}

    # Grid search hyperparameter tuning
    model_nr = 1
    hp = list(hp_dict.keys())
    for hp_0 in hp_dict[hp[0]]:
        for hp_1 in hp_dict[hp[1]]:
            for hp_2 in hp_dict[hp[2]]:
                for hp_3 in hp_dict[hp[3]]:   
                    # Define the name for the run
                    run_name = f"{timestamp}/grid_search/model_{model_nr}/layer_{hp_0}_activaten_{hp_1}_batch_{hp_2}_dr_{hp_3}"

                    # Build the model archtecture
                    tabular_model = model_architecture(lr= 1e-3, dropout=hp_3, batch_size=hp_2, activation=hp_1, layers=hp_0, experiment_path=experiment_path, run_name=run_name)

                    # Train and evaluate the model
                    start = time.time()
                    tabular_model.fit(train=df_train, validation=df_val)
                    stop = time.time()
                    training_time = int(stop - start)

                    # Get training and validation results
                    model_path = os.path.join(experiment_path, run_name) 
                    model_path = os.path.join(model_path, "version_1")
                    event_path = find_event_file(model_path)
                    train_val_results_dict = get_model_results(event_path)
                    
                    # Get predictions on the testset
                    pred_df = tabular_model.predict(df_test)
                    test_results = get_metrics_results(y_true=pred_df["Relevant fuer Messung"], y_pred=pred_df["prediction"])
                    
                    # Get the index of the best iteration 
                    metric = config["earlystopping_metric"]
                    if config["early_stopping_mode"] == "min":
                        best_iteration = train_val_results_dict[metric].index(min(train_val_results_dict[metric]))
                    else:
                        best_iteration = train_val_results_dict[metric].index(max(train_val_results_dict[metric]))

                    # Add model and results to list of trained models
                    str_model_nr = "model_nr " + str(model_nr)
                    trained_models[str_model_nr] = []
                    trained_models[str_model_nr].append(tabular_model)
                    trained_models[str_model_nr].append(train_val_results_dict)
                    trained_models[str_model_nr].append(test_results)
                    trained_models[str_model_nr].append(pred_df)
                    trained_models[str_model_nr].append(best_iteration)

                    # Store the model results in a dataframe
                    df_gridsearch.loc[model_nr, "Model"] = model_nr
                    df_gridsearch.loc[model_nr, "Train loss"] = round(train_val_results_dict["train_loss"][best_iteration], 5)
                    df_gridsearch.loc[model_nr, "Train accuracy"] = round(train_val_results_dict["train_accuracy"][best_iteration], 5)
                    df_gridsearch.loc[model_nr, "Train f1-score"] = round(train_val_results_dict["train_f1_score"][best_iteration], 5)
                    df_gridsearch.loc[model_nr, "Validation loss"] = round(train_val_results_dict["valid_loss"][best_iteration], 5)
                    df_gridsearch.loc[model_nr, "Validation accuracy"] = round(train_val_results_dict["valid_accuracy"][best_iteration], 5)
                    df_gridsearch.loc[model_nr, "Validation f1-score"] = round(train_val_results_dict["valid_f1_score"][best_iteration], 5)
                    df_gridsearch.loc[model_nr, "Test accuracy"] = round(test_results["test_accuracy"], 5)
                    df_gridsearch.loc[model_nr, "Test sensitivity"] = round(test_results["test_sensitivity"], 5)
                    df_gridsearch.loc[model_nr, "Test f1-score"] = round(test_results["test_f1_score"], 5)                  
                    df_gridsearch.loc[model_nr, "Training time (s)"] = training_time
                    df_gridsearch.loc[model_nr, "Trained epochs"] = int(train_val_results_dict["epoch"][-1])
                    df_gridsearch.loc[model_nr, hp_columns[0]] = hp_0
                    df_gridsearch.loc[model_nr, hp_columns[1]] = hp_1
                    df_gridsearch.loc[model_nr, hp_columns[2]] = hp_2
                    df_gridsearch.loc[model_nr, hp_columns[3]] = hp_3

                    model_nr = model_nr + 1
                    break
                break
            break
        break
    
    # Store the results of the trained models after grid search
    model_folder = os.path.join(experiment_path, f"{timestamp}")
    df_gridsearch.to_csv(os.path.join(model_folder, "grid_search/results_grid_search.csv"))

    # Sort the results by the validation f1-score in descening order 
    df_gridsearch_sorted = df_gridsearch.sort_values(by=["Validation f1-score"], ascending=False)

    return trained_models, model_folder, df_gridsearch_sorted

def k_fold_crossvalidation(df_train_cv, df_gridsearch_sorted, experiment_path, timestamp):
    
    # Declare the column names to generate the dataframe which stores the results and the hyperparameter of the models using grid search
    fix_columns = ["Model", "avg train loss", "avg train accuracy", "avg train f1_score", "avg validation loss", "avg validation accuracy", "avg validation f1_score"]
    hp_columns = list(config["hyperparameter"].keys())
    columns = fix_columns + hp_columns

    # Create empty dataframe with the predefined column names
    results_cv = pd.DataFrame(columns=columns)

    # Get the number of models which should be used for cross-validation
    top_x_models = math.ceil(df_gridsearch_sorted.shape[0] * config["top_x_models_for_cv"])

    # Set the number of folds 
    number_of_folds = config["k-folds"]

    # Total number of folds
    total_cv_models = number_of_folds * top_x_models

    # Set StratifiedKFold
    kfold = StratifiedKFold(n_splits=config["k-folds"], shuffle=True, random_state=42)

    # Declare empty lists to store the training and validation results
    train_loss = []
    train_accuracy = []
    train_f1_score = []
    valid_loss = []
    valid_accuracy = []
    valid_f1_score = []

    # Cross-Validation
    logger.info(f"Start to validate the top {top_x_models} models by using {number_of_folds}-fold cross-validation... ")
    for i in range(top_x_models):
        # Get the hyperparameter according to the sorted list (F1-Score descending order) of results after grid search hyperparameter tuning 
        dropout = df_gridsearch_sorted.iloc[i]["dropout"]
        batch = df_gridsearch_sorted.iloc[i]["batch_size"]
        activation = df_gridsearch_sorted.iloc[i]["activation_functions"]
        layers = df_gridsearch_sorted.iloc[i]["layers"]

        # K-Fold Cross-Validation for each fold
        logger.info(f"Start {number_of_folds}-fold cross-validation for model {i+1}/{total_cv_models}:")
        for fold, (train_idx, val_idx) in enumerate(kfold.split(df_train_cv, df_train_cv[config["target"]])):
            logger.info(f"Fold: {fold+1}/{number_of_folds}")
            
            # Define the name for the run
            run_name = f"{timestamp}/cross_validation/model_{i+1}/version{fold+1}_layer_{layers}_activaten_{activation}_batch_{batch}_dr_{dropout}"

            # Get the train and validaiton data for the current fold
            train_fold = df_train_cv.iloc[train_idx]
            val_fold = df_train_cv.iloc[val_idx]

            # Initialize the tabular model
            tabular_model = model_architecture(lr= 1e-3, dropout=dropout, batch_size=batch, activation=activation, layers=layers, experiment_path=experiment_path, run_name=run_name)

            # Fit the model
            tabular_model.fit(train=train_fold, validation=val_fold)
            
            # Get results after training
            model_path = os.path.join(experiment_path, run_name) 
            model_path = os.path.join(model_path, "version_1")
            event_path = find_event_file(model_path)
            train_val_results_dict = get_model_results(event_path)

            # Get index of best iteration 
            metric = config["earlystopping_metric"]
            if config["early_stopping_mode"] == "min":
                best_iteration = train_val_results_dict[metric].index(min(train_val_results_dict[metric]))
            else:
                best_iteration = train_val_results_dict[metric].index(max(train_val_results_dict[metric]))

            train_loss.append(train_val_results_dict["train_loss"][best_iteration])
            train_accuracy.append(train_val_results_dict["train_accuracy"][best_iteration])
            train_f1_score.append(train_val_results_dict["train_f1_score"][best_iteration])
            valid_loss.append(train_val_results_dict["valid_loss"][best_iteration])
            valid_accuracy.append(train_val_results_dict["valid_accuracy"][best_iteration])
            valid_f1_score.append(train_val_results_dict["valid_f1_score"][best_iteration])
        
        results_cv.loc[i, "Model"] = df_gridsearch_sorted.iloc[i]["Model"]
        results_cv.loc[i, "avg train loss"] = round(mean(train_loss), 6)
        results_cv.loc[i, "avg train accuracy"] = round(mean(train_accuracy), 6)
        results_cv.loc[i, "avg train f1_score"] = round(mean(train_f1_score), 6)
        results_cv.loc[i, "avg validation loss"] = round(mean(valid_loss), 6)
        results_cv.loc[i, "avg validation accuracy"] = round(mean(valid_accuracy), 6)
        results_cv.loc[i, "avg validation f1_score"] = round(mean(valid_f1_score), 6)
        results_cv.loc[i, "layers"] = layers
        results_cv.loc[i, "dropout"] = dropout
        results_cv.loc[i, "batch_size"] = batch
        results_cv.loc[i, "activation_functions"] = activation
    
    model_folder = os.path.join(experiment_path, f"{timestamp}")
    results_cv.to_csv(os.path.join(model_folder, "cross_validation/results_cross_validation.csv"))

    results_cv_sorted = results_cv.sort_values(by=["avg validation f1_score"], ascending=False)
    best_model_index = results_cv_sorted.iloc[0]["Model"]
        
    logger.success("Cross-Validation was successfull!")

    return best_model_index, results_cv_sorted

def train():
    dateTimeObj = datetime.now()
    timestamp = dateTimeObj.strftime("%Y%m%d_%H%M")
    experiment_path = Path(f"src/deep_learning_pipeline/trained_models")

    # Load data
    data_folder = config["folder_processed_dataset"]
    X_train, y_train, X_val, y_val, X_test, y_test, df_preprocessed, df_train, df_val, df_test, weight_factor = load_training_data(data_folder=data_folder, binary_model=True)

    # Keep only relevant features
    relevant_columns = config["continuous_cols"] + config["categorical_cols"] + config["target"]
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
    
    trained_models, model_folder, df_gridsearch_sorted = grid_search(df_train, df_val, df_test, experiment_path, timestamp)
    
    df_train_cv = pd.concat([df_train, df_val], ignore_index=True)

    best_model_index, results_cv_sorted = k_fold_crossvalidation(df_train_cv, df_gridsearch_sorted, experiment_path, timestamp)

    model_name = "model_nr " + str(best_model_index)
    best_model_after_crossvalidation = trained_models[model_name][0]

    # Store the best model after hyperparameter tuning and crossvalidation
    binary_model_path = os.path.join(model_folder, "binary_model")
    best_model_after_crossvalidation.save_model(binary_model_path)

    best_iteration = trained_models[model_name][4]

    # Store the loss plot
    train_loss = trained_models[model_name][1]["train_loss"]
    valid_loss = trained_models[model_name][1]["valid_loss"]
    store_metrics(best_iteration=best_iteration, train_results=train_loss, validation_results=valid_loss, metric="Loss", model_folder_path=binary_model_path)

    # Store the accuracy plot
    train_accuracy = trained_models[model_name][1]["train_accuracy"]
    valid_accuracy = trained_models[model_name][1]["valid_accuracy"]
    store_metrics(best_iteration=best_iteration, train_results=train_accuracy, validation_results=valid_accuracy, metric="Accuracy", model_folder_path=binary_model_path)

    # Store the f1-score plot
    train_f1_score = trained_models[model_name][1]["train_f1_score"]
    valid_f1_score = trained_models[model_name][1]["valid_f1_score"]
    store_metrics(best_iteration=best_iteration, train_results=train_f1_score, validation_results=valid_f1_score, metric="F1-Score", model_folder_path=binary_model_path)  

    pred_df = best_model_after_crossvalidation.predict(df_test)
    # Store wrong predictions on the testset
    df_filtered = pred_df[pred_df["Relevant fuer Messung"] != pred_df["prediction"]]
    df_filtered.to_excel(os.path.join(binary_model_path, "wrong_predictions.xlsx"))

    # Store confusion matrix
    store_confusion_matrix(df_test=pred_df["Relevant fuer Messung"], df_pred=pred_df["prediction"], model_folder_path=binary_model_path)

    # Train and store the final model
    df_train_final_model = pd.concat([df_train, df_test], ignore_index=True)
    dropout = results_cv_sorted.iloc[0]["dropout"]
    batch = results_cv_sorted.iloc[0]["batch_size"]
    activation = results_cv_sorted.iloc[0]["activation_functions"]
    layers = results_cv_sorted.iloc[0]["layers"]
    run_name = f"{timestamp}/final_binary_model/layer_{layers}_activaten_{activation}_batch_{batch}_dr_{dropout}"
    final_model = model_architecture(lr= 1e-3, dropout=dropout, batch_size=batch, activation=activation, layers=layers, experiment_path=experiment_path, run_name=run_name)
    final_model.fit(train=df_train_final_model, validation=df_val)

    # Get results after training
    model_path = os.path.join(experiment_path, run_name) 
    model_path = os.path.join(model_path, "version_1")
    event_path = find_event_file(model_path)
    train_val_results_dict = get_model_results(event_path)

    # Get index of best iteration 
    metric = config["earlystopping_metric"]
    if config["early_stopping_mode"] == "min":
        best_iteration = train_val_results_dict[metric].index(min(train_val_results_dict[metric]))
    else:
        best_iteration = train_val_results_dict[metric].index(max(train_val_results_dict[metric]))

    train_loss = train_val_results_dict["train_loss"][best_iteration]
    train_accuracy = train_val_results_dict["train_accuracy"][best_iteration]
    train_f1_score = train_val_results_dict["train_f1_score"][best_iteration]
    valid_loss = train_val_results_dict["valid_loss"][best_iteration]
    valid_accuracy = train_val_results_dict["valid_accuracy"][best_iteration]
    valid_f1_score = train_val_results_dict["valid_f1_score"][best_iteration]

    final_model_path = os.path.join(experiment_path, f"{timestamp}/final_binary_model/model_1")

    final_model.save_model(final_model_path)

    # Create logfile
    create_logfile(final_model_path, data_folder, train_loss, train_accuracy, train_f1_score, valid_loss, valid_accuracy, valid_f1_score, dropout, layers, activation, batch)

    # Store the loss plot
    store_metrics(best_iteration=best_iteration, train_results=train_val_results_dict["train_loss"], validation_results=train_val_results_dict["valid_loss"], metric="Loss", model_folder_path=final_model_path)

    # Store the accuracy plot
    store_metrics(best_iteration=best_iteration, train_results=train_val_results_dict["train_accuracy"], validation_results=train_val_results_dict["valid_accuracy"], metric="Accuracy", model_folder_path=final_model_path)

    # Store the f1-score plot
    store_metrics(best_iteration=best_iteration, train_results=train_val_results_dict["train_f1_score"], validation_results=train_val_results_dict["valid_f1_score"], metric="F1-Score", model_folder_path=final_model_path)  

    # Store the model summary to pkl file
    #tabular_model.summary()

# %%
if __name__ == "__main__":
    
    train()

