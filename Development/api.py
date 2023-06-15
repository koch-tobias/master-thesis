from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
import pandas as pd
from pyxlsb import open_workbook as open_xlsb
from Prepare_data import prepare_and_add_labels
from typing import Annotated, Union
from io import BytesIO
import pickle
import numpy as np
from Feature_Engineering import preprocess_dataset
from boundingbox_calculations import find_valid_space
from config import general_params, train_settings, api_setting
import os
from loguru import logger

class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

app = FastAPI()

@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=404,
        content={"message": f"{exc.name}"},
    )

def get_model(folder_path):
    # Load model for relevance
    for file in os.listdir(folder_path):
        if file.startswith("model"):
            model_path =  os.path.join(folder_path, file)

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

    return lgbm, vectorizer, vocabulary

def get_X(vocab, vectorizer, df):
    # Convert the vocabulary list to a dictionary
    vocabulary_dict = {word: index for index, word in enumerate(vocab)}

    # Set the vocabulary of the vectorizer to the loaded vocabulary
    vectorizer.vocabulary_ = vocabulary_dict
    X = vectorizer.transform(df['Benennung (bereinigt)']).toarray()

    # Combine text features with other features
    if train_settings["use_only_text"] == False:
        X = np.concatenate((X, df[general_params["features_for_model"]].values), axis=1)
    
    return X

def dataframe_to_dict(df):
    result_dict = {}
    for index, row in df.iterrows():
        sachnummer = row['Sachnummer']
        benennung = row['Benennung (dt)']
        einheitsname = row['Einheitsname']
        ausfuehrung = row["Linke/Rechte Ausfuehrung"]

        result_dict[sachnummer] = [benennung, einheitsname]
        if (ausfuehrung == 'Linke Ausfuehrung') or (ausfuehrung == 'Rechte Ausfuehrung'):
            result_dict[sachnummer].append(ausfuehrung)

    return result_dict    

def identify_relevent_parts(dataframes):
    
    try:
        df, ncars = prepare_and_add_labels(dataframes)
    except:
        raise UnicornException(name=f"Preparing the dataframe failed!")

    try:
        trainset = pd.read_excel("../data/preprocessed_data/df_preprocessed.xlsx")
    except:
        raise UnicornException(name=f"file (df_preprocessed) not found!")
    
    try:
        trainset_relevant_parts = trainset[trainset["Relevant fuer Messung"] == "Ja"]
        trainset_relevant_parts = trainset_relevant_parts[(trainset_relevant_parts['X-Min_transf'] != 0) & (trainset_relevant_parts['X-Max_transf'] != 0)]    
        unique_names = trainset_relevant_parts["Einheitsname"].unique().tolist()
        unique_names.sort()
    except:
        raise UnicornException(name=f"Getting unique names failed!")

    try:
        lgbm_binary, vectorizer_binary, vocabulary_binary = get_model(api_setting["model_binary"])
        lgbm_multiclass, vectorizer_multiclass, vocabulary_multiclass = get_model(api_setting["model_multiclass"])
    except:
        raise UnicornException(name=f"Loading the models failed!")

    for i in range(len(df)):
        try:
            df_preprocessed, df_for_plot = preprocess_dataset(df[i], cut_percent_of_front=0.20)
        except:
            raise UnicornException(name=f"Data preprocessing failed!")

        try:
            logger.info("Start using the Machine Learning model to identify the relevant car part... ")
            X_binary = get_X(vocabulary_binary, vectorizer_binary, df_preprocessed)
            probs_binary = lgbm_binary.predict_proba(X_binary)
            y_pred_binary = np.where(probs_binary[:, 1] > 0.7, 1, 0)

            X_multiclass = get_X(vocabulary_multiclass, vectorizer_multiclass, df_preprocessed)
            probs_multiclass = lgbm_multiclass.predict_proba(X_multiclass)
            y_pred_multiclass = probs_multiclass.argmax(axis=1)
            logger.success("Done!")
        except:
            raise UnicornException(name=f"Label transformation failed!")


        try:
            # Load the LabelEncoder
            with open(api_setting["model_multiclass"] + '/label_encoder.pkl', 'rb') as f:
                le = pickle.load(f) 
        except:
            raise UnicornException(name=f"Loading Label Encoder failed!")

        y_pred_multiclass_names = le.inverse_transform(y_pred_multiclass) 

        df_preprocessed = df_preprocessed.reset_index(drop=True)
        for index, row in df_preprocessed.iterrows():
            if y_pred_binary[index] == 1: 
                df_preprocessed.loc[index,'Relevant fuer Messung'] = 'Ja'
            else:
                df_preprocessed.loc[index,'Relevant fuer Messung'] = 'Nein'

            df_preprocessed.loc[index,'Einheitsname'] = y_pred_multiclass_names[index]
            df_preprocessed.loc[index,'Wahrscheinlichkeit Relevanz'] = probs_binary[:, 1][index]
            df_preprocessed.loc[index,'Wahrscheinlichkeit Einheitsname'] = probs_multiclass[index, y_pred_multiclass[index]]

        logger.info("Check if identified car parts are in bounding-box space... ")
        df_preprocessed = df_preprocessed[df_preprocessed['Relevant fuer Messung'] == 'Ja']
        einheitsname_not_found = []
        try:
            for index, row in df_preprocessed.iterrows():
                for name in unique_names:
                    trainset_name = trainset_relevant_parts[(trainset_relevant_parts["Einheitsname"] == name)].reset_index(drop=True)
                    corners, _, _, _ = find_valid_space(trainset_name)
                    x_min = np.min(corners[:, 0])
                    x_max = np.max(corners[:, 0])
                    y_min = np.min(corners[:, 1])
                    y_max = np.max(corners[:, 1])
                    z_min = np.min(corners[:, 2])
                    z_max = np.max(corners[:, 2])
                    valid_volume_min = trainset_name["volume"].min()
                    valid_volume_max = trainset_name["volume"].max()
                    
                    if ((row["X-Min_transf"] == 0) and (row["X-Max_transf"] == 0)):
                        df_preprocessed.loc[index,'In Bounding-Box-Position von'] = 'No Bounding-Box information'
                    else:
                        df_preprocessed.loc[index,'In Bounding-Box-Position von'] = 'None'
                        if ((row["X-Min_transf"] > x_min) and (row["X-Max_transf"] < x_max)):
                            if ((row["Y-Min_transf"] > y_min) and (row["Y-Max_transf"] < y_max)): 
                                    if ((row["Z-Min_transf"] > z_min) and (row["Z-Max_transf"] < z_max)):
                                        if ((row["volume"] >= valid_volume_min*0.9) and (row["volume"] <= valid_volume_max*1.1)):
                                            df_preprocessed.loc[index,'In Bounding-Box-Position von'] = name
                                            if (row["Wahrscheinlichkeit Relevanz"] > 0.95) and ((row["Einheitsname"] == "Dummy")):
                                                df_preprocessed.loc[index,'Einheitsname'] = name
                                            break
        except:
            raise UnicornException(name=f"Bounding-Box comparison failed!")
        
        logger.success("Done!")


    try:
        for name in unique_names:        
            if name not in df_preprocessed['Einheitsname'].unique():
                einheitsname_not_found.append(name)
    except:
        raise UnicornException(name=f"Searching not found Einheitsnamen failed!")

    try:
        df_preprocessed = df_preprocessed.reset_index(drop=True)
        df_preprocessed = df_preprocessed.loc[:,["Sachnummer", "Benennung (dt)", "Einheitsname", "L/R-Kz."]]
        df_preprocessed["L/R-Kz."] = df_preprocessed["L/R-Kz."].fillna(' ')
        df_preprocessed.rename(columns={'L/R-Kz.':'Linke/Rechte Ausfuehrung'}, inplace=True)
        df_preprocessed.loc[df_preprocessed['Einheitsname'] == "Dummy", 'Einheitsname'] = 'Kein Einheitsname gefunden'
        df_preprocessed.loc[df_preprocessed['Linke/Rechte Ausfuehrung'] == "L", 'Linke/Rechte Ausfuehrung'] = 'Linke Ausfuehrung'
        df_preprocessed.loc[df_preprocessed['Linke/Rechte Ausfuehrung'] == "R", 'Linke/Rechte Ausfuehrung'] = 'Rechte Ausfuehrung'
    except:
        raise UnicornException(name=f"Modifying columns failed!")

    try:
        df_json = df_preprocessed.to_dict(orient='series') 
    except:
        raise UnicornException(name=f"Converting the dataframe to a dict failed!")
    
    df_json = dataframe_to_dict(df_preprocessed)
    
    return df_json

@app.post("/api/get_relevant_parts/")
async def post_relevant_parts(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_excel(BytesIO(contents))
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    dataframes = []
    dataframes.append(df)

    df_json = identify_relevent_parts(dataframes)
    
    return df_json


@app.get("/get_relevant_parts/{file_path:path}")
async def get_relevant_parts(file_path: str):
    try:
        df = pd.read_excel(file_path, header=None, skiprows=1)
    except:
        raise UnicornException(name=f"Load Excel to dataframe failed!")

    df.columns = df.iloc[0]
    df = df.iloc[1:]
    dataframes = []
    dataframes.append(df)

    df_json = identify_relevent_parts(dataframes)
   
    return df_json