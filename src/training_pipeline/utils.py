# %%
import pandas as pd
import numpy as np

import os
from loguru import logger
import pickle

from src.config import paths, general_params

def get_model(folder_path):
    final_model_path = ""
    pre_model_path = ""

    # Load model for relevance
    for file in os.listdir(folder_path):
        if file.startswith("final"):
            final_model_path =  os.path.join(folder_path, file)
        elif file.startswith("model"):
            pre_model_path =  os.path.join(folder_path, file)

    if final_model_path != "":
        model_path = final_model_path
    else:
        model_path = pre_model_path

    with open(model_path, "rb") as fid:
        model = pickle.load(fid)

    # Load the vectorizer from the file
    vectorizer_path = folder_path + "/vectorizer.pkl"
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    # Get the vocabulary of the training data
    vocab_path = folder_path + "/vocabulary.pkl"
    with open(vocab_path, 'rb') as f:
        vocabulary = pickle.load(f) 

    bbox_features_path = folder_path + "/boundingbox_features.pkl"
    with open(bbox_features_path, 'rb') as f:
        bbox_features = pickle.load(f)  

    return model, vectorizer, vocabulary, bbox_features

def get_X(vocab, vectorizer, bbox_features, df_preprocessed):
    # Convert the vocabulary list to a dictionary
    vocabulary_dict = {word: index for index, word in enumerate(vocab)}

    # Set the vocabulary of the vectorizer to the loaded vocabulary
    vectorizer.vocabulary_ = vocabulary_dict
    X = vectorizer.transform(df_preprocessed['Benennung (bereinigt)']).toarray()

    # Combine text features with other features
    if general_params["use_only_text"] == False:      
        X = np.concatenate((X, df_preprocessed[bbox_features].values), axis=1)
    
    return X

# %%
def load_dataset(binary_model: bool):
    data_folder = paths["folder_processed_dataset"]
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
def store_trained_model(model, best_iteration, val_auc, hp, index_best_model, model_folder_path, finalmodel):
    # save model
    if finalmodel:
        model_path = model_folder_path + f"final_model_{str(val_auc)[2:6]}_validation_auc.pkl"
    else:
        model_path = model_folder_path + f"model_{str(val_auc)[2:6]}_validation_auc.pkl"

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
        log_text = "Trainings dataset path: {}\n".format(paths["folder_processed_dataset"])
        f= open(model_folder_path + "logging.txt","w+")
        f.write(log_text)
        f.write("Validation AUC: {}\n".format(val_auc))
        f.write("Trained Iterations: {}\n".format(best_iteration))
        f.write("Model index in GridSearch Hyperparametertuning: {}\n".format(index_best_model))
        f.write("Hyperparameter:\n")
        for key in hp:
            f.write("{}: {}\n".format(key, hp[key]))
        f.close()

# %%
def store_predictions(y_test, y_pred, probs, df_preprocessed, df_test, model_folder_path, binary_model):

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