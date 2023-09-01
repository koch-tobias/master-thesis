# Used to train the models
import pandas as pd
import numpy as np

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from loguru import logger
from pathlib import Path
import warnings
import os
import shutil
import time
from datetime import datetime
import math
import pickle
from statistics import mean

import yaml
from yaml.loader import SafeLoader
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

import sys
sys.path.append(os.getcwd())

from classifier import Classifier
from plot_functions import Visualization
from src.deployment.classification import Identifier 
from evaluation import evaluate_model, get_best_metric_results, store_predictions
from src.utils import store_trained_model, load_training_data

warnings.filterwarnings("ignore")

def model_fit(X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array, weight_factor: float, hp_in_iteration: dict, binary_model: bool, method: str):
    ''' 
    Fits and trains a model based on the specified parameters.
    Args:
        X_train (pd.DataFrame): Training set features
        y_train (pd.Series): Training set target values
        X_val (pd.DataFrame): Validation set features
        y_val (pd.Series): Validation set target values
        weight_factor (float): Weight factor applied to handle class imbalance in the target variable
        hp_in_iteration (dict): Dictionary of hyperparameters with their values to be used in each training iteration
        binary_model (bool): Boolean value indicating whether the model is binary or multiclass
        method (str): Machine learning method used for modeling (lgbm, xgboost or catboost)
    Returns:
        Tuple: A tuple containing the trained model, evaluation metrics and the chosen evaluation metric used for training and validation
    '''
    evals = {}
    if method == "lgbm":
        callbacks = [lgb.early_stopping(config["train_settings"]["early_stopping"], verbose=5), lgb.record_evaluation(evals)]
    elif method == "xgboost":
        if binary_model:
            metric_name = config["xgb_params_binary"]["metrics"][0]
        else:
            metric_name = config["xgb_params_multiclass"]["metrics"][0]
        callbacks = [xgb.callback.EarlyStopping(config["train_settings"]["early_stopping"], metric_name=metric_name)]

    if binary_model:
        model, metrics = Classifier.binary_classifier(weight_factor=weight_factor, hp=hp_in_iteration, method=method)
    else:
        model, metrics = Classifier.multiclass_classifier(weight_factor=weight_factor, hp=hp_in_iteration, method=method)
    
    if method == "lgbm":
        model.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)], 
                eval_metric=metrics,
                callbacks=callbacks
                )

    elif method == "xgboost":
        model.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)], 
                callbacks=callbacks
                )      

        evals = model.evals_result()    
          
    elif method == "catboost":
        model.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)], 
                silent=True
                )

        evals = model.get_evals_result()

    return model, evals, metrics

def get_model_folder_path(binary_model: bool, folder_path: Path) -> str:
    ''' 
    This function returns the model folder path depending on the value of binary_model. If binary_model is True, then the folder path with "Binary_model/" appended will be returned, otherwise the folder path with "Multiclass_model/" appended will be returned. 
    If the folder path does not exist, it will be created.
    Args:
        binary_model (bool): Boolean value that determines whether the model is binary (True) or multiclass (False).
        folder_path (str): The base directory where the model folder will be created.
    Return:
        model_folder_path (str): The model folder path where the model files will be stored.
    '''
    if binary_model:
        model_folder_path = os.path.join(folder_path, "Binary_model/")
    else:
        model_folder_path = os.path.join(folder_path, "Multiclass_model/")

    try:
        os.makedirs(model_folder_path)
    except FileExistsError:
        print(f"{model_folder_path}, already exist!")
        pass

    return model_folder_path

def fit_eval_model(X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array, X_test: np.array, y_test: np.array, weight_factor: float or dict, hp_in_iteration: dict, df: pd.DataFrame, model_results_dict: dict, num_models_trained: int, total_models: int, binary_model: bool, method: str) -> tuple[dict, pd.DataFrame, dict]:
    ''' 
    This function trains and evaluates a model based on the given data. It takes training, validation, and testing data as input and returns a dictionary of different evaluation metrics for each trained model, along with a dataframe containing details for each iteration of the model training process.
    Args:
        X_train: Training data features.
        y_train: Training data labels.
        X_val: Validation data features.
        y_val: Validation data labels.
        X_test: Testing data features.
        y_test: Testing data labels.
        weight_factor: A weight factor for the loss function.
        hp_in_iteration: The number of hyperparameter combinations to try in each iteration.
        df: A Pandas dataframe that contains details for each iteration of the model training process.
        model_results_dict: A dictionary of different evaluation metrics for each trained model.
        num_models_trained: The number of models already trained.
        total_models: The total number of models to train.
        binary_model: A boolean value that indicates whether the model is binary or multiclass.
        method: The method used for feature selection.
    Return:
        model_results_dict: A dictionary of different evaluation metrics for each trained model.
        df: A Pandas dataframe that contains details for each iteration of the model training process.
        metrics: A dictionary of different evaluation metrics for the trained model.
    '''

    start = time.time()
    gbm, evals, metrics = model_fit(X_train, y_train, X_val, y_val, weight_factor, hp_in_iteration, binary_model, method)
    stop = time.time()
    training_time = int(stop - start)
    y_pred, probs, test_accuracy, test_sensitivity, val_auc, val_loss, train_auc, train_loss, df_new = evaluate_model(gbm, X_test, y_test, evals, hp_in_iteration, num_models_trained, training_time, df.columns, binary_model=binary_model, method=method)    
    df = pd.concat([df, df_new])
    logger.info(f"Modell {num_models_trained+1}/{total_models} trained with a evaluation accuracy = {val_auc} and with a evaluation loss = {val_loss}")
    model_results_dict["models_list"].append(gbm)
    model_results_dict["evals_list"].append(evals)
    model_results_dict["test_sensitivity_results"].append(test_sensitivity)
    model_results_dict["y_pred_list"].append(y_pred)
    model_results_dict["test_accuracy_results"].append(test_accuracy)
    model_results_dict["val_auc_results"].append(val_auc)
    model_results_dict["val_loss_results"].append(val_loss)
    model_results_dict["train_auc_results"].append(train_auc)
    model_results_dict["train_loss_results"].append(train_loss)
    model_results_dict["probs_list"].append(probs)
    num_models_trained = num_models_trained + 1
    return model_results_dict, df, metrics

def create_result_df(hp_dict: dict) -> tuple[pd.DataFrame, int]:
    ''' 
    This function creates a dataframe to store the results of the model training process. The dataframe contains columns for the model name, as well as columns for different evaluation metrics, including training and validation auc and loss, test accuracy and sensitivity, and training time. It also takes a dictionary of hyperparameters as input.
    Args:
        hp_dict (dict): A dictionary of hyperparameters to try.
    Return:
        df (Pandas dataframe): A dataframe to store the results of the model training process.
        total_models (int): An integer representing the total number of models that will be trained using this dataframe.
    '''
    fix_columns = ["model_name", "train auc", "train loss", "validation auc", "validation loss", "test accuracy", "test sensitivity", "test f1_score", "Training Time (s)"]

    total_models = 1
    for hp in hp_dict:
        total_models = total_models * len(hp_dict[hp])
    
    columns = fix_columns + list(hp_dict.keys())
    df = pd.DataFrame(columns=columns)

    return df, total_models

def grid_search(X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array, X_test: np.array, y_test: np.array, weight_factor: float or dict, hp_dict: dict, binary_model: bool, method: str) -> tuple[pd.DataFrame, dict, dict]:
    ''' 
    This function performs model training with grid search hyperparameter tuning. It takes training, validation, and testing data as input, along with a dictionary of hyperparameters to try, as well as a boolean value indicating whether the model is binary or multiclass, and a method used for feature selection. It returns a dataframe containing the results of the model training process, as well as a dictionary of different evaluation metrics for each trained model.
    Args:
        X_train: Training data features.
        y_train: Training data labels.
        X_val: Validation data features.
        y_val: Validation data labels.
        X_test: Testing data features.
        y_test: Testing data labels.
        weight_factor: A weight factor for the loss function.
        hp_dict: A dictionary of hyperparameters to try.
        binary_model: A boolean value indicating whether the model is binary or multiclass.
        method: The method used for feature selection.
    Return:
        df (Pandas dataframe): A dataframe containing the results of the model training process.
        model_results_dict (dict): A dictionary of different evaluation metrics for each trained model.
        metrics (dict): A dictionary of evaluation metrics for the trained model.
    '''

    logger.info("Start training with grid search hyperparameter tuning...")

    model_results_dict = {"models_list": [], "evals_list": [], "test_sensitivity_results": [], "test_accuracy_results": [], "val_auc_results": [], "val_loss_results": [], "train_auc_results": [], "train_loss_results": [], "y_pred_list": [], "probs_list": []}
    num_models_trained = 0

    df, total_models = create_result_df(hp_dict)
    hp = list(hp_dict.keys())

    for hp_0 in hp_dict[hp[0]]:
        for hp_1 in hp_dict[hp[1]]:
            for hp_2 in hp_dict[hp[2]]:
                for hp_3 in hp_dict[hp[3]]:
                    hp_in_iteration = {hp[0]: hp_0, hp[1]: hp_1, hp[2]: hp_2, hp[3]: hp_3}
                    model_results_dict, df, metrics = fit_eval_model(X_train, y_train, X_val, y_val, X_test, y_test, weight_factor, hp_in_iteration, df, model_results_dict, num_models_trained, total_models, binary_model, method)
                    num_models_trained = num_models_trained + 1
                    break
                break
            break
        break

    logger.success("Grid search hyperparameter tuning was successfull!")

    return df, model_results_dict, metrics


def get_fold_data(X: np.array, y: np.array, train_index: int, val_index: int) -> tuple[np.array]:
    ''' 
    This function takes the training and validation sets for a specific fold as input, along with the indices of the training and validation sets, and returns the features and labels for the training and validation sets for that fold.
    Args:
        X: The feature data.
        y: The label data.
        train_index: The indices of the training data for the current fold.
        val_index: The indices of the validation data for the current fold.
    Return:
        X_train_fold: The feature data for the training set for the current fold.
        X_val_fold: The feature data for the validation set for the current fold.
        y_train_fold: The label data for the training set for the current fold.
        y_val_fold: The label data for the validation set for the current fold.
    '''

    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    return X_train_fold, X_val_fold, y_train_fold, y_val_fold


def k_fold_crossvalidation(X: np.array, y: np.array, X_test: np.array, y_test: np.array, weight_factor: float or dict, df: pd.DataFrame, model_results_dict: dict, folder_path: Path, method: str, binary_model: bool) -> pd.DataFrame:
    ''' 
    This function performs k-fold cross-validation for the top x models generated by grid search hyperparameter tuning. 
    It takes training, validation, and testing data as input, along with a weight factor for the loss function, a dataframe containing the results of the grid search hyperparameter tuning, a dictionary of different evaluation metrics for each trained model, the folder path to store the models and results, the method used for feature selection, and a boolean value indicating whether the model is binary or multiclass. 
    It returns a dataframe containing the average evaluation metrics for each of the top x models based on cross-validation.
    Args:
        X: Training data features.
        y: Training data labels.
        X_test: Testing data features.
        y_test: Testing data labels.
        weight_factor: A weight factor for the loss function.
        df: A dataframe containing the results of the grid search hyperparameter tuning.
        model_results_dict: A dictionary of different evaluation metrics for each trained model.
        folder_path: The folder path to store the models and results.
        method: The method used for feature selection.
        binary_model: A boolean value indicating whether the model is binary or multiclass.
    Return:
        df_cv (Pandas dataframe): A dataframe containing the average evaluation metrics for each of the top x models based on cross-validation.
    '''
    kfold = StratifiedKFold(n_splits=config["train_settings"]["k-folds"], shuffle=True, random_state=config["dataset_params"]["seed"])

    fix_columns = ["model_name", "avg train auc", "avg train loss", "avg validation auc", "avg validation loss", "avg test accuracy", "avg test sensitivity"]

    if method == "lgbm":
        hp_columns = list(config["lgbm_hyperparameter"].keys())
    elif method == "xgboost":
        hp_columns = list(config["xgb_hyperparameter"].keys())
    elif method == "catboost":
        hp_columns = list(config["cb_hyperparameter"].keys())

    columns = fix_columns + hp_columns
    df_cv = pd.DataFrame(columns=columns)

    top_x_models = math.ceil(df.shape[0] * config["train_settings"]["top_x_models_for_cv"])
    number_of_folds = config["train_settings"]["k-folds"]
    total_cv_models = number_of_folds * top_x_models
    count_trained_models = 0

    logger.info(f"Start to validate the top {top_x_models} models by using {number_of_folds}-fold cross-validation... ")
    for i in range(top_x_models):
        cv_results_dict = {"models_list": [], "evals_list": [], "test_sensitivity_results": [], "test_accuracy_results": [], "val_auc_results": [], "val_loss_results": [], "train_auc_results": [], "train_loss_results": [], "y_pred_list": [], "probs_list": []}

        logger.info(f"Start {number_of_folds}-fold cross-validation for model {i+1}:")
        for train_index, val_index in kfold.split(X, y):
            
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = get_fold_data(X=X, y=y, train_index=train_index, val_index=val_index)        

            hp_dict = {hp_columns[0]: df[hp_columns[0]].iloc[i], hp_columns[1]: df[hp_columns[1]].iloc[i], hp_columns[2]: df[hp_columns[2]].iloc[i], hp_columns[3]: df[hp_columns[3]].iloc[i]}

            cv_results_dict, df_temp, metrics = fit_eval_model(X_train=X_train_fold, y_train=y_train_fold, X_val=X_val_fold, y_val=y_val_fold, X_test=X_test, y_test=y_test, weight_factor=weight_factor, hp_in_iteration=hp_dict, df=df, model_results_dict=cv_results_dict, num_models_trained=count_trained_models, total_models=total_cv_models, binary_model=binary_model, method=method)
            count_trained_models = count_trained_models + 1

        avg_val_auc = round(mean(cv_results_dict["val_auc_results"]), 6)
        avg_val_loss = round(mean(cv_results_dict["val_loss_results"]), 6)
        avg_train_auc = round(mean(cv_results_dict["train_auc_results"]), 6)
        avg_train_loss = round(mean(cv_results_dict["train_loss_results"]), 6)
        avg_test_sensitivity = round(mean(cv_results_dict["test_sensitivity_results"]), 6)
        avg_test_auc = round(mean(cv_results_dict["test_accuracy_results"]), 6)
        for key in list(hp_dict.keys()):
            df_cv.loc[i, key] = df[key].iloc[i]
        df_cv.loc[i, "model_name"] = df["model_name"].iloc[i]
        df_cv.loc[i, "avg validation auc"] = avg_val_auc
        df_cv.loc[i, "avg validation loss"] = avg_val_loss      
        df_cv.loc[i, "avg train auc"] = avg_train_auc
        df_cv.loc[i, "avg train loss"] = avg_train_loss     
        df_cv.loc[i, "avg test accuracy"] = avg_test_auc
        df_cv.loc[i, "avg test sensitivity"] = avg_test_sensitivity
        df_cv.loc[i, "early stopping (iterations)"] = int(config["train_settings"]["early_stopping"])
        df_cv.loc[i, "index"] = int(df.index[i])

    logger.success("Cross-Validation was successfull!")

    return df_cv

def copy_labelencoder(model_folder_path: Path):
    data_folder = Path(config["train_settings"]["folder_processed_dataset"])
    src = os.path.join(data_folder, "label_encoder.pkl") 
    dst = os.path.join(model_folder_path, "label_encoder.pkl")
    shutil.copy2(src, dst)

def train_model(folder_path: Path, binary_model: bool, method: str):
    ''' 
    This function trains a machine learning model using the specified method and hyperparameters with a hold-out method and grid search hyperparameter tuning. 
    It then performs k-fold cross-validation on the 10% best models, selects the best model, and stores its results. 
    It also trains a new model on the entire dataset with the selected hyperparameters and stores its results.
    Args:
        folder_path (str): the path to the folder containing the dataset
        binary_model (bool): a flag indicating whether the model is binary or not
        method (str): the machine learning algorithm to use for training
    Return: None
    '''
    global model_folder_path 
    model_folder_path = get_model_folder_path(binary_model=binary_model, folder_path=folder_path)

    if method == "lgbm":
        hp_dict = config["lgbm_hyperparameter"]
    elif method == "xgboost":
        hp_dict = config["xgb_hyperparameter"]
    elif method == "catboost":
        hp_dict = config["cb_hyperparameter"]
   
    # Training with the hold out method and grid search hyperparameter tuning
    data_folder = Path(config["train_settings"]["folder_processed_dataset"])
    X_train, y_train, X_val, y_val, X_test, y_test, df_preprocessed, df_train, df_val, df_test, weight_factor = load_training_data(data_folder, binary_model=binary_model)

    # Copy the label encoder from the dataset path to the model folder. This file is used in the deployment process.
    copy_labelencoder(model_folder_path)

    # Store the list of uniform names (for the xAi pipeline)
    unique_names = df_preprocessed[config['labels']['multiclass_column']].unique().tolist()
    with open(os.path.join(model_folder_path, "list_of_uniform_names.pkl"), 'wb') as f:
        pickle.dump(unique_names, f)

    # Start grid search hyperparamter tuning
    df, model_results_dict, metrics = grid_search(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, weight_factor=weight_factor, hp_dict=hp_dict, binary_model=binary_model, method=method)

    df.to_csv(os.path.join(model_folder_path, "hyperparametertuning_results.csv"))

    # K-Fold Cross-Validation with the 5 best model trained with the hold out method 
    df = df.sort_values(by=["validation auc"], ascending=False)

    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)

    df_cv = k_fold_crossvalidation(X=X, y=y, X_test=X_test, y_test=y_test, weight_factor=weight_factor, df=df, model_results_dict=model_results_dict, folder_path=folder_path, method=method, binary_model=binary_model)

    df_cv.to_csv(os.path.join(model_folder_path, "crossvalidation_results.csv"))
    df_cv = df_cv.sort_values(by=["avg validation auc"], ascending=False)

    index_best_model = int(df_cv["index"].iloc[0])

    logger.info("Start storing the best model with metrics plots and wrong predictions according to the validation auc...")
    best_model = model_results_dict["models_list"][index_best_model]

    best_iteration = Identifier.get_best_iteration(best_model, method)

    if method == 'lgbm':
        Visualization.store_metrics(evals=model_results_dict["evals_list"][index_best_model], best_iteration=best_iteration, model_folder_path=model_folder_path, binary_model=binary_model, finalmodel=False)
    else:
        Visualization.plot_metric_custom(evals=model_results_dict["evals_list"][index_best_model], best_iteration=best_iteration, model_folder_path=model_folder_path, method=method, binary=binary_model, finalmodel=False)

    hp_keys = list(hp_dict.keys())
    best_hp = {}
    for key in hp_keys:
        best_hp[key] = df[key].iloc[index_best_model]

    store_trained_model(model=best_model, metrics=metrics, best_iteration=best_iteration, val_auc=df_cv["avg validation auc"].iloc[0], hp=best_hp, index_best_model=index_best_model, model_folder_path=model_folder_path, finalmodel=False)
    y_pred, probs, best_iteration = Identifier.model_predict(model=best_model, X_test=X_test, method=method, binary_model=binary_model)
    Visualization.store_confusion_matrix(y_test=y_test, y_pred=y_pred, folder_path=folder_path, model_folder_path=model_folder_path, binary_model=binary_model)
    store_predictions(y_test=y_test, y_pred=y_pred, probs=probs, df_preprocessed=df_preprocessed, df_test=df_test, model_folder_path=model_folder_path, binary_model=binary_model)
    logger.success("Storing was successfull!")

    logger.info("Train a new model on the entire data with the parameters of the best identified model...")
    X_train = np.concatenate((X_train, X_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)
        
    gbm_final, evals_final, metrics = model_fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, weight_factor=weight_factor, hp_in_iteration=best_hp, binary_model=binary_model, method=method)
    best_iteration = Identifier.get_best_iteration(model=gbm_final, method=method)
    _, _, val_auc, _ = get_best_metric_results(evals=evals_final, best_iteration=best_iteration, method=method, binary_model=binary_model)
    store_trained_model(model=gbm_final, metrics=metrics, best_iteration=best_iteration, val_auc=val_auc, hp=best_hp, index_best_model=index_best_model, model_folder_path=model_folder_path, finalmodel=True)
    if method == 'lgbm':
        Visualization.store_metrics(evals=evals_final, best_iteration=best_iteration, model_folder_path=model_folder_path, binary_model=binary_model, finalmodel=True)
    else:
        Visualization.plot_metric_custom(evals=evals_final, best_iteration=best_iteration, model_folder_path=model_folder_path, method=method, binary=binary_model, finalmodel=True)

    logger.success("Training the model on the entire data was successfull!")

def main():
    train_binary_model = config["train_settings"]["train_binary_model"]
    train_multiclass_model = config["train_settings"]["train_multiclass_model"]

    method = config["train_settings"]["ml-method"]

    dateTimeObj = datetime.now()
    timestamp = dateTimeObj.strftime("%d%m%Y_%H%M")

    folder_path = Path(f"src/training_pipeline/trained_models/{method}_HyperparameterTuning_{timestamp}/")
    
    if train_binary_model:
        logger.info("Start training the binary models...")
        train_model(folder_path, binary_model=True, method=method)

    if train_multiclass_model:
        logger.info("Start training the multiclass models...")
        train_model(folder_path, binary_model=False, method=method)
    
if __name__ == "__main__":
    
    main()