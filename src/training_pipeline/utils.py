# %%
import pandas as pd
import numpy as np

import os
import pickle
import json
from loguru import logger

import yaml
from yaml.loader import SafeLoader
with open('../config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# %%
def load_dataset(binary_model: bool):
    ''' 
    This function loads the preprocessed dataset along with train, validation and test sets from the specified data folder. The binary_model parameter indicates whether the dataset is for a binary classification task or not.
    Args:
        binary_model: A boolean value indicating whether the dataset is for a binary classification task or not.
    Return:
        X_train: An array or DataFrame containing the training set features.
        y_train: An array or DataFrame containing the training set labels.
        X_val: An array or DataFrame containing the validation set features.
        y_val: An array or DataFrame containing the validation set labels.
        X_test: An array or DataFrame containing the test set features.
        y_test: An array or DataFrame containing the test set labels.
        df_preprocessed: The preprocessed dataset.
        df_test: A DataFrame containing the test set data.
        weight_factor: A float value indicating the weight factor for the dataset.
    '''
    data_folder = config["paths"]["folder_processed_dataset"]
    path_trainset = os.path.join(data_folder, "processed_dataset.csv")  

    if os.path.exists(path_trainset):
        df_preprocessed = pd.read_csv(path_trainset) 
    else:
        logger.error(f"No trainset found! Please check if the dataset exist at following path: {path_trainset}. If not, please use the file generate.py to create the processed dataset.")

    if binary_model:
        data_folder = data_folder + "binary/"
        train_val_test_path = os.path.join(data_folder, "binary_train_test_val_split.pkl")
        train_val_test_df_paths = os.path.join(data_folder, "binary_train_test_val_dataframes.pkl")
    else:
        data_folder = data_folder + "multiclass/"
        train_val_test_path = os.path.join(data_folder, "multiclass_train_test_val_split.pkl")
        train_val_test_df_paths = os.path.join(data_folder, "multiclass_train_test_val_dataframes.pkl")

    with open(train_val_test_path, 'rb') as handle:
        train_val_test_dict = pickle.load(handle)

    with open(train_val_test_df_paths, 'rb') as handle:
        train_val_test_df_dict = pickle.load(handle)
    
    X_train = train_val_test_dict["X_train"]
    y_train = train_val_test_dict["y_train"]
    X_val = train_val_test_dict["X_val"]
    y_val = train_val_test_dict["y_val"]
    X_test = train_val_test_dict["X_test"]
    y_test = train_val_test_dict["y_test"]
    weight_factor = train_val_test_dict["weight_factor"]

    df_test = train_val_test_df_dict["df_test"].reset_index(drop=True)

    return X_train, y_train, X_val, y_val, X_test, y_test, df_preprocessed, df_test, weight_factor

# %%
def store_trained_model(model, metrics: str, best_iteration: int, val_auc: float, hp: dict, index_best_model: int, model_folder_path: str, finalmodel: bool) -> None:
    ''' 
    This function stores the trained model, hyperparameters, metrics and best iteration information in a pickled file at the provided model folder path and logs the validation AUC and training information in a txt file.
    Args:
        model: The trained model.
        metrics: A dictionary containing the evaluation metrics and their values for the model.
        best_iteration: An integer indicating the number of iterations the model took to converge to the best solution.
        val_auc: A float representing the AUC score of the validation set.
        hp: A dictionary containing the hyperparameters used to train the model.
        index_best_model: An integer indicating the index of the best model in the hyperparameter tuning process.
        model_folder_path: A string indicating the path where the trained model and logs will be saved.
        finalmodel: A boolean value indicating whether the model to be saved is the final model.
    Return: None
    '''
    # save model
    if finalmodel:
            model_path = model_folder_path + f"final_model.pkl"
    else:
            model_path = model_folder_path + f"model.pkl"

    with open(model_path, "wb") as filestore:
        pickle.dump(model, filestore)

    logging_file_path = model_folder_path + "logging.txt"
    if os.path.isfile(logging_file_path):
        log_text = "\nValidation AUC (final model): {}\n".format(val_auc)
        f= open(model_folder_path + "logging.txt","a")
        f.write("\n_________________________________________________\n")
        f.write(log_text)
        f.write("Trained Iterations: {}\n".format(best_iteration))
        f.close()
    else:
        dataset_path = "Dataset: {}\n".format(config["paths"]["folder_processed_dataset"])
        model_folder = "Model folder path: {}\n".format(model_folder_path)
        f= open(model_folder_path + "logging.txt","w+")
        f.write(dataset_path)
        f.write(model_folder)
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

# %%
def store_predictions(y_test: np.arry, y_pred: np.arry, probs: np.arry, df_preprocessed: pd.DataFrame, df_test: pd.DataFrame, model_folder_path: str, binary_model: bool) -> None:
    ''' 
    This function saves the predicted outputs and related information in a CSV file called wrong_predictions.csv at the provided model folder path.
    Args:
        y_test: An array or DataFrame containing the true labels of the test set.
        y_pred: An array or DataFrame containing the predicted labels of the test set.
        probs: An array or DataFrame containing the predicted probabilities for each label of the test set.
        df_preprocessed: The preprocessed dataset.
        df_test: A DataFrame containing the test set data.
        model_folder_path: A string indicating the path where the prediction results will be saved.
        binary_model: A boolean value indicating whether the dataset is for a binary classification task or not.
    Return: None
    '''

    if binary_model:
        class_names = df_preprocessed['Relevant fuer Messung'].unique()
    else:
        class_names = df_preprocessed["Einheitsname"].unique()
        class_names = sorted(class_names)

    df_wrong_predictions = pd.DataFrame(columns=['Sachnummer', 'Benennung (dt)', 'Derivat', 'Predicted', 'True', 'Probability'])

    try:
        y_test = y_test.to_numpy()
    except:
        pass

    # Ausgabe der Vorhersagen, der Wahrscheinlichkeiten und der wichtigsten Features
    for i in range(len(y_test)):
        try:
            if y_pred[i] != y_test[i]:
                df_wrong_predictions.loc[i,"Sachnummer"] = df_test.loc[i, "Sachnummer"]
                df_wrong_predictions.loc[i,"Benennung (dt)"] = df_test.loc[i, "Benennung (dt)"]
                df_wrong_predictions.loc[i,"Derivat"] = df_test.loc[i, "Derivat"]
                df_wrong_predictions.loc[i,"Predicted"] = class_names[y_pred[i]]
                df_wrong_predictions.loc[i,"True"] = class_names[y_test[i]]
                if binary_model:
                    if probs[i][1] >= 0.5:
                        df_wrong_predictions.loc[i,"Probability"] = probs[i][1]
                    else:
                        df_wrong_predictions.loc[i,"Probability"] = 1 - probs[i][1]
                else:
                    df_wrong_predictions.loc[i,"Probability"] = probs[i][1]
        except:
            pass
        
    # Serialize data into file:
    df_wrong_predictions.to_csv(model_folder_path + "wrong_predictions.csv")