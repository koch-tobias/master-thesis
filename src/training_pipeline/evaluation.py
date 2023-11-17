import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, recall_score, f1_score, fbeta_score, precision_score

from loguru import logger


from pathlib import Path
import os
import yaml
from yaml.loader import SafeLoader
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

import sys
sys.path.append(os.getcwd())

from src.deployment.inference import Identifier
    
def get_best_metric_results(evals: dict, best_iteration: int, method: str, binary_model: bool) -> tuple[float]:
    ''' 
    The function is used to get the best metric results from a given model evaluation during the training process. 
    Args:
        evals: a dictionary containing evaluation results obtained during the training process
        best_iteration: an integer representing the best iteration number of the model
        method: a string indicating the model method (lightgbm, xgboost or catboost)
        binary_model: a boolean indicating whether the model is a binary classification model (True) or not (False)
    Return:
        train_auc: the training set AUC (Area Under Curve) score of the best iteration
        train_loss: the training set loss of the best iteration
        val_auc: the validation set AUC score of the best iteration
        val_loss: the validation set loss of the best iteration
    '''
    valid_name = "validation_1"
    training_name = "validation_0"

    if method == "lgbm":
        if binary_model:
            loss = config["lgbm_params_binary"]["loss"] 
            val_metric = 'fbeta'
        else:
            loss = config["lgbm_params_multiclass"]["loss"] 
            #val_metric = 'auc_mu'

    elif method == "xgboost":
        if binary_model:
            loss = config["xgb_params_binary"]["loss"] 
            val_metric = 'xgb_custom_fbeta_score'
        else:
            loss = config["xgb_params_multiclass"]["loss"] 

    elif method == "catboost":
        if binary_model:
            val_metric = 'F:beta=2'
            loss = config["cb_params_binary"]["loss"] 
        else:
            '''
            if config["cb_params_multiclass"]["metric"] == 'AUC':
                val_metric = 'AUC:type=Mu'
            else:
                val_metric = config["cb_params_multiclass"]["metric"]
            '''
            loss = config["cb_params_multiclass"]["loss"]    

    if binary_model:
        val_fbeta = evals[valid_name][val_metric][best_iteration]
        train_fbeta = evals[training_name][val_metric][best_iteration]
    else:
        val_fbeta = 0
        train_fbeta = 0

    val_loss = evals[valid_name][loss][best_iteration]
    train_loss = evals[training_name][loss][best_iteration]

    return train_fbeta, train_loss, val_fbeta, val_loss

def store_predictions(y_test: np.array, y_pred: np.array, probs: np.array, df_preprocessed: pd.DataFrame, df_test: pd.DataFrame, model_folder_path: Path, binary_model: bool) -> None:
    ''' 
    Stores wrong predictions, true labels, predicted labels, probabilities, and additional information based on the input parameters in a CSV file.
    Args:
        y_test: true labels
        y_pred: predicted labels
        probs: probability values
        df_preprocessed: preprocessed DataFrame
        df_test: test DataFrame
        model_folder_path: path to store the wrong_predictions.csv file
        binary_model: bool, indicates whether binary classification or multiclass classification is expected
    Return: None 
    '''
    if binary_model:
        class_names = ["Relevant", "Not relevant"]
    else:
        df_only_relevants = df_preprocessed[df_preprocessed["Relevant fuer Messung"] == "Ja"]
        class_names = df_only_relevants["Einheitsname"].unique()
        class_names = sorted(class_names)

    df_wrong_predictions = pd.DataFrame(columns=['Sachnummer', 'Benennung (dt)', 'Derivat', 'Predicted', 'True', 'Probability'])

    try:
        y_test = y_test.to_numpy()
    except:
        pass
    
    # Store all wrongly identified car parts in a csv file to analyse the model predictions
    for i in range(len(y_test)):
        if y_pred[i] != y_test[i]:
            df_wrong_predictions.loc[i,"Sachnummer"] = df_test.loc[i, "Sachnummer"]
            df_wrong_predictions.loc[i,"Benennung (dt)"] = df_test.loc[i, "Benennung (dt)"]
            df_wrong_predictions.loc[i,"Derivat"] = df_test.loc[i, "Derivat"]
            df_wrong_predictions.loc[i,"Predicted"] = class_names[y_pred[i]]
            df_wrong_predictions.loc[i,"True"] = class_names[y_test[i]]
            if binary_model:
                if probs[i][1] >= config["prediction_settings"]["prediction_threshold"]:
                    df_wrong_predictions.loc[i,"Probability"] = probs[i][1]
                else:
                    df_wrong_predictions.loc[i,"Probability"] = 1 - probs[i][1]
            else:
                df_wrong_predictions.loc[i,"Probability"] = probs[i][1]

    df_wrong_predictions.to_csv(os.path.join(model_folder_path, "wrong_predictions.csv"))

def calculate_metrics(y, y_pred, binary):

    if binary:
        sensitivity = recall_score(y, y_pred)
        precision = precision_score(y, y_pred)
        fbeta = fbeta_score(y, y_pred, beta=2)
    else:
        fbeta = fbeta_score(y, y_pred, beta=2, average='weighted')
        sensitivity = recall_score(y, y_pred, average='weighted')
        precision = precision_score(y, y_pred, average='weighted')

    return precision, sensitivity, fbeta


def evaluate_model(model, X_test: np.array, y_test: np.array, X_val: np.array, y_val: np.array, X_train: np.array, y_train: np.array, evals: dict, hp_in_iteration: dict, num_models_trained: int, training_time: float, df_columns: list, binary_model: bool, method: str) -> tuple[np.array, np.array, float, float, float, float, float, float, pd.DataFrame]:
    ''' 
    This function evaluates a trained machine learning model. It does this by predicting the test set labels and probabilities.
    Args:
        model: Trained machine learning model
        X_test: Feature set of test set
        y_test: Target values of test set
        evals: Evaluation results obtained during the training process
        hp_in_iteration: Hyperparameters that were used to train the model
        num_models_trained: Integer which indicates how many models have been trained before the current model
        training_time: Time in seconds that the current model spent in the training and evaluation process
        df_columns: List of the column names in the dataframe
        binary_model: A boolean which indicates whether the model is binary or multiclass
        method: A string which indicates the machine learning library which was used to train the model
    Return:
        y_pred: predicted labels for the test set
        probs: probability estimates for the predicted labels
        accuracy: accuracy score for the test set
        sensitivity: sensitivity score for the test set
        val_auc: Validation set's area under curve (ROC-AUC) score of the best model iteration
        val_loss: Validation set's log loss score of the best model iteration
        train_auc: Train set's area under curve (ROC-AUC) score of the best model iteration
        train_loss: Train set's log loss score of the best model iteration
        df_new: a pandas dataframe containing the results of the evaluation process.
    '''
    y_pred_train, probs_train, best_iteration_train  = Identifier.model_predict(model=model, X_test=X_train, method=method, binary_model=binary_model)
    y_pred_val, probs_val, best_iteration_val  = Identifier.model_predict(model=model, X_test=X_val, method=method, binary_model=binary_model)
    y_pred_test, probs_test, best_iteration_test  = Identifier.model_predict(model=model, X_test=X_test, method=method, binary_model=binary_model)

    fbeta_train, train_loss, fbeta_val, val_loss = get_best_metric_results(evals=evals, best_iteration=best_iteration_train, method=method, binary_model=binary_model)
    
    if binary_model:
        precision_val, sensitivity_val, _ = calculate_metrics(y_val, y_pred_val, binary_model)
    else:
        precision_train, sensitivity_train, fbeta_train = calculate_metrics(y_train, y_pred_train, binary_model)
        precision_val, sensitivity_val, fbeta_val = calculate_metrics(y_val, y_pred_val, binary_model)
    
    precision_test, sensitivity_test, fbeta_test = calculate_metrics(y_test, y_pred_test, binary_model)

    df_new = pd.DataFrame(columns=df_columns)

    df_new.loc[num_models_trained, "model_name"] = f"model_{str(fbeta_val)[2:6]}"
    df_new.loc[num_models_trained, "train loss"] = train_loss
    df_new.loc[num_models_trained, "train fbeta_score"] = fbeta_train
    df_new.loc[num_models_trained, "validation loss"] = val_loss
    df_new.loc[num_models_trained, "validation precision"] = precision_val
    df_new.loc[num_models_trained, "validation sensitivity"] = sensitivity_val
    df_new.loc[num_models_trained, "validation fbeta_score"] = fbeta_val
    df_new.loc[num_models_trained, "test sensitivity"] = sensitivity_test
    df_new.loc[num_models_trained, "test precision"] = precision_test
    df_new.loc[num_models_trained, "test fbeta_score"] = fbeta_test
    df_new.loc[num_models_trained, "Training Time (s)"] = training_time 
    df_new.loc[num_models_trained, "patience"] = int(config["train_settings"]["early_stopping"])
    df_new.loc[num_models_trained, "Trained iterations"] = int(best_iteration_train)
    for hp in hp_in_iteration:
        df_new.loc[num_models_trained, hp] = hp_in_iteration[hp]

    return train_loss, fbeta_train, val_loss, precision_val, sensitivity_val, fbeta_val, sensitivity_test, precision_test, fbeta_test, df_new