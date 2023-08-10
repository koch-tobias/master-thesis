# %%
import pandas as pd
import numpy as np

import pickle

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from src.deployment_pipeline.prediction import model_predict

import yaml
from yaml.loader import SafeLoader
with open('../config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

#%%
def add_feature_importance(model, model_folder_path: str) -> pd.DataFrame:
    ''' 
    This function is used to extract the most important features from a given model. 
    It takes in a trained model and the path of the folder where the model vocabulary is stored. The output of the function is a pandas DataFrame containing the column names, corresponding features, and their importance scores.
    Args:
        model: a trained model object
        model_folder_path: a string representing the path to the folder where the model vocabulary is stored
    Return: 
        df_features: a pandas DataFrame containing the column names, corresponding features, and their importance scores.
    '''
    vocabulary_path = model_folder_path + "vocabulary.pkl"
    # Get the vocabulary of the training data
    with open(vocabulary_path, 'rb') as f:
        vocabulary = pickle.load(f)

    # Extrahieren der wichtigsten Features
    boost = model.booster_
    importance = boost.feature_importance()
    column = boost.feature_name()
    feature_dict = {vocabulary.shape[0]+index: key for index, key in enumerate(config["general_params"]["features_for_model"])}

    path_store_features = model_folder_path + "features.xlsx"

    df_features = pd.DataFrame(columns=['Column','Feature','Importance Score'])
    df_features["Column"] = column
    df_features["Importance Score"] = importance
    for j in range(len(column)):
        if j < vocabulary.shape[0]:
            df_features.loc[j,"Feature"] = vocabulary[j]
        else:
            df_features.loc[j,"Feature"] = feature_dict[j]

    #df_features.to_excel(path_store_features)
    return df_features

# %%
def get_features(model_folder_path: str) -> tuple[list, list]:
    ''' 
    This function is used to retrieve the list of features and their importance scores from a given path of a folder where the features are stored in an Excel file. 
    The function returns two lists - a complete list of feature names and a list of top 20 most important features.
    Args:
        model_folder_path: a string representing the path of the folder where the features are stored
    Return:
        feature_list: a list of strings representing all features stored in the Excel file
        topx_important_features: a list of integers representing the indices of the 20 most important features in the feature_list. If the Excel file does not exist, a message indicating the error is printed and the function returns None.
    '''
    path_store_features = model_folder_path + "features.xlsx"
    try:
        df_features = pd.read_excel(path_store_features) 
        topx_important_features = df_features.sort_values(by=["Importance Score"], ascending=False).head(20)
        topx_important_features = topx_important_features.index.tolist()
        feature_list = df_features["Feature"].values.tolist()

        return feature_list, topx_important_features
    except:
        print(f"Error: File {path_store_features} does not exist!")
    
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
    if method == "lgbm":
        valid_name = "valid_1"
        training_name = "training"
        if binary_model:
            auc = config["lgbm_params_binary"]["metrics"][0] 
            loss = config["lgbm_params_binary"]["metrics"][1] 
        else:
            auc = config["lgbm_params_multiclass"]["metrics"][0] 
            loss = config["lgbm_params_multiclass"]["metrics"][1] 

    elif method == "xgboost":
        valid_name = "validation_1"
        training_name = "validation_0"
        if binary_model:
            auc = config["xgb_params_binary"]["metrics"][0]
            loss = config["xgb_params_binary"]["metrics"][1] 
        else:
            auc = config["xgb_params_multiclass"]["metrics"][0]
            loss = config["xgb_params_multiclass"]["metrics"][1]

    elif method == "catboost":
        valid_name = "validation_1"
        training_name = "validation_0"
        if binary_model:
            auc = config["cb_params_binary"]["metrics"][0]
            loss = config["cb_params_binary"]["metrics"][1] 
        else:
            if config["cb_params_multiclass"]["metrics"][0] == 'AUC':
                auc = 'AUC:type=Mu'
            else:
                auc = config["cb_params_multiclass"]["metrics"][0]
            loss = config["cb_params_multiclass"]["metrics"][1]        

    val_auc = evals[valid_name][auc][best_iteration]
    val_loss = evals[valid_name][loss][best_iteration]
    train_auc = evals[training_name][auc][best_iteration]
    train_loss = evals[training_name][loss][best_iteration]
    return train_auc, train_loss, val_auc, val_loss

# %%
def evaluate_model(model, X_test: np.array, y_test: np.array, evals: dict, hp_in_iteration: dict, num_models_trained: int, training_time: float, df_columns: list, binary_model: bool, method: str) -> tuple[np.array, np.array, float, float, float, float, float, float, pd.DataFrame]:
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
    
    y_pred, probs, best_iteration  = model_predict(model=model, X_test=X_test, method=method, binary_model=binary_model)

    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, average='macro')

    df_new = pd.DataFrame(columns=df_columns)

    train_auc, train_loss, val_auc, val_loss = get_best_metric_results(evals=evals, best_iteration=best_iteration, method=method, binary_model=binary_model)

    df_new.loc[num_models_trained, "model_name"] = f"model_{str(val_auc)[2:6]}"
    for hp in hp_in_iteration:
        df_new.loc[num_models_trained, hp] = hp_in_iteration[hp]
    df_new.loc[num_models_trained, "train auc"] = train_auc
    df_new.loc[num_models_trained, "train loss"] = train_loss
    df_new.loc[num_models_trained, "validation auc"] = val_auc
    df_new.loc[num_models_trained, "validation loss"] = val_loss
    df_new.loc[num_models_trained, "test accuracy"] = accuracy
    df_new.loc[num_models_trained, "test sensitivity"] = sensitivity
    df_new.loc[num_models_trained, "early stopping (iterations)"] = int(config["train_settings"]["early_stopping"])
    df_new.loc[num_models_trained, "Training Time (s)"] = training_time 

    return y_pred, probs, accuracy, sensitivity, val_auc, val_loss, train_auc, train_loss, df_new