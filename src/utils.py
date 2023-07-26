# %%
import pandas as pd
import numpy as np

import os
import shutil
from loguru import logger
import pickle

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from data_pipeline.data_analysis import analyse_data_split
from data_pipeline.feature_engineering import vectorize_data
from training_pipeline.config import train_settings, general_params, paths

# %%
def train_test_val(df, use_only_text, model_folder_path, binary_model):
    logger.info("Generate train, validation and test split...")
    X = vectorize_data(df, model_folder_path)

    # Combine text features with other features
    features = general_params["features_for_model"]
    bbox_features_dict = {"features_for_model": features}
    with open(model_folder_path + 'boundingbox_features.pkl', 'wb') as fp:
        pickle.dump(bbox_features_dict, fp)

    if use_only_text == False:
        X = np.concatenate((X, df[features].values), axis=1)

    if binary_model:
        y = df['Relevant fuer Messung']
        y = y.map({'Ja': 1, 'Nein': 0})
    else:
        y = df['Einheitsname']
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        with open(model_folder_path + 'label_encoder.pkl', 'wb') as f: 
            pickle.dump(le, f)  

    weight_factor = get_weight_factor(y, df, binary_model)     

    indices = np.arange(X.shape[0])

    X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(X, y, indices, test_size=train_settings["val_size"], stratify=y, random_state=42)
    X_val, X_test, y_val, y_test, indices_val, indices_test = train_test_split(X_val, y_val, indices_val, test_size=train_settings["test_size"], stratify=y_val, random_state=42)

    df_test = df.iloc[indices_test]

    logger.success("Train, validation and test sets are generated!")

    return X_train, y_train, X_val, y_val, X_test, y_test, df_test, weight_factor

# %%
def get_weight_factor(y, df, binary_model):
    if binary_model == False:
        # Get list of unique values in column "Einheitsname"
        unique_einheitsnamen = np.unique(y)
        weight_factor = {}
        for name in unique_einheitsnamen:
            weight_factor[name] = round(np.count_nonzero(y != name) / np.count_nonzero(y == name))
            if weight_factor[name] == 0:
                weight_factor[name] = 1
    else:    
        weight_factor = round(df[df["Relevant fuer Messung"]=="Nein"].shape[0] / df[df["Relevant fuer Messung"]=="Ja"].shape[0])

    return weight_factor

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
        lgbm = pickle.load(fid)

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

    return lgbm, vectorizer, vocabulary, bbox_features

def get_X(vocab, vectorizer, bbox_features, df_preprocessed):
    # Convert the vocabulary list to a dictionary
    vocabulary_dict = {word: index for index, word in enumerate(vocab)}

    # Set the vocabulary of the vectorizer to the loaded vocabulary
    vectorizer.vocabulary_ = vocabulary_dict
    X = vectorizer.transform(df_preprocessed['Benennung (bereinigt)']).toarray()

    # Combine text features with other features
    if train_settings["use_only_text"] == False:      
        X = np.concatenate((X, df_preprocessed[bbox_features].values), axis=1)
    
    return X

# %%
def load_dataset(model_folder_path, binary_model: bool):

    path_trainset = paths["processed_dataset"]

    if os.path.exists(path_trainset):
        df_preprocessed = pd.read_excel(path_trainset) 
    else:
        logger.error(f"No trainset found! Please check if the dataset exist at following path: {path_trainset}. If not, please use the file generate.py to create the processed dataset.")

    X_train, y_train, X_val, y_val, X_test, y_test, df_test, weight_factor = train_test_val(df_preprocessed, model_folder_path=model_folder_path, binary_model=binary_model)

    analyse_data_split(df_preprocessed, y_train, y_val, y_test, model_folder_path, binary_model)

    return X_train, y_train, X_val, y_val, X_test, y_test, df_preprocessed, df_test, weight_factor

# %%
def store_trained_model(model, val_auc, index_best_model, model_folder_path):
    # save model
    if val_auc != -1:
        model_path = model_folder_path + f"model{index_best_model}_{str(val_auc)[2:6]}_validation_auc.pkl"
    else:
        model_path = model_folder_path + f"final_model.pkl"

    with open(model_path, "wb") as filestore:
        pickle.dump(model, filestore)

    src_path = "models/df_trainset.xlsx"
    dst_path = model_folder_path + "df_trainset.xlsx"
    shutil.copy(src_path, dst_path)