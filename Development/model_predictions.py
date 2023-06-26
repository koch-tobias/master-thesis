import pandas as pd
import numpy as np
import pickle
from loguru import logger
from Data_Preprocessing import prepare_and_add_labels, preprocess_dataset, get_model, get_X
from boundingbox_calculations import find_valid_space
from config import model_paths_api, paths_api, model_paths, paths

def predict_on_new_data(df, use_api: bool):

    df, ncar = prepare_and_add_labels(df)
    if use_api:
        trainset = pd.read_excel(model_paths_api["model_folder"] + "/df_trainset.xlsx")
    else:
        trainset = pd.read_excel(model_paths["model_folder"] + "/df_testset.xlsx")
    trainset_relevant_parts = trainset[trainset["Relevant fuer Messung"] == "Ja"]
    trainset_relevant_parts = trainset_relevant_parts[(trainset_relevant_parts['X-Min_transf'] != 0) & (trainset_relevant_parts['X-Max_transf'] != 0)]    
    unique_names = trainset_relevant_parts["Einheitsname"].unique().tolist()
    unique_names.sort()

    logger.info(unique_names)
    
    if use_api:
        lgbm_binary, vectorizer_binary, vocabulary_binary = get_model(model_paths_api["lgbm_binary"])
        lgbm_multiclass, vectorizer_multiclass, vocabulary_multiclass = get_model(model_paths_api["lgbm_multiclass"])
    else:
        lgbm_binary, vectorizer_binary, vocabulary_binary = get_model(model_paths["lgbm_binary"])
        lgbm_multiclass, vectorizer_multiclass, vocabulary_multiclass = get_model(model_paths["lgbm_multiclass"])       

    df_preprocessed, df_for_plot = preprocess_dataset(df, cut_percent_of_front=0.20)

    X_binary = get_X(vocabulary_binary, vectorizer_binary, df_preprocessed)
    probs_binary = lgbm_binary.predict_proba(X_binary)
    y_pred_binary = np.where(probs_binary[:, 1] > 0.7, 1, 0)

    X_multiclass = get_X(vocabulary_multiclass, vectorizer_multiclass, df_preprocessed)
    probs_multiclass = lgbm_multiclass.predict_proba(X_multiclass)
    y_pred_multiclass = probs_multiclass.argmax(axis=1)

    # Load the LabelEncoder
    if use_api:
        with open(model_paths_api["lgbm_multiclass"] + '/label_encoder.pkl', 'rb') as f:
            le = pickle.load(f) 
    else:
        with open(model_paths["lgbm_multiclass"] + '/label_encoder.pkl', 'rb') as f:
            le = pickle.load(f) 

    y_pred_multiclass_names = le.inverse_transform(y_pred_multiclass) 

    df_relevant_parts = df_preprocessed.reset_index(drop=True)
    for index, row in df_relevant_parts.iterrows():
        if y_pred_binary[index] == 1: 
            df_relevant_parts.loc[index,'Relevant fuer Messung'] = 'Ja'
        else:
            df_relevant_parts.loc[index,'Relevant fuer Messung'] = 'Nein'

        df_relevant_parts.loc[index,'Einheitsname'] = y_pred_multiclass_names[index]
        df_relevant_parts.loc[index,'Wahrscheinlichkeit Relevanz'] = probs_binary[:, 1][index]
        df_relevant_parts.loc[index,'Wahrscheinlichkeit Einheitsname'] = probs_multiclass[index, y_pred_multiclass[index]]

    df_relevant_parts = df_relevant_parts[df_relevant_parts['Relevant fuer Messung'] == 'Ja']
    einheitsname_not_found = []
    for index, row in df_relevant_parts.iterrows():
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
                df_relevant_parts.loc[index,'In Bounding-Box-Position von'] = 'No Bounding-Box information'
            else:
                df_relevant_parts.loc[index,'In Bounding-Box-Position von'] = 'None'
                if ((row["X-Min_transf"] > x_min) and (row["X-Max_transf"] < x_max)):
                    if ((row["Y-Min_transf"] > y_min) and (row["Y-Max_transf"] < y_max)): 
                            if ((row["Z-Min_transf"] > z_min) and (row["Z-Max_transf"] < z_max)):
                                if ((row["volume"] >= valid_volume_min*0.9) and (row["volume"] <= valid_volume_max*1.1)):
                                    df_relevant_parts.loc[index,'In Bounding-Box-Position von'] = name
                                    if (row["Wahrscheinlichkeit Relevanz"] > 0.95) and ((row["Einheitsname"] == "Dummy")):
                                        df_relevant_parts.loc[index,'Einheitsname'] = name
                                    break

    for name in unique_names:        
        if name not in df_relevant_parts['Einheitsname'].unique():
            einheitsname_not_found.append(name)

    df_relevant_parts = df_relevant_parts.reset_index(drop=True)
    df_relevant_parts["L/R-Kz."] = df_relevant_parts["L/R-Kz."].fillna(' ')
    #df_preprocessed.rename(columns={'L/R-Kz.':'Linke/Rechte Ausfuehrung'}, inplace=True)
    df_relevant_parts.loc[df_relevant_parts['Einheitsname'] == "Dummy", 'Einheitsname'] = 'Kein Einheitsname gefunden'
    df_relevant_parts.loc[df_relevant_parts["L/R-Kz."] == "L", "L/R-Kz."] = 'Linke Ausfuehrung'
    df_relevant_parts.loc[df_relevant_parts["L/R-Kz."] == "R", "L/R-Kz."] = 'Rechte Ausfuehrung'

    return df_preprocessed, df_relevant_parts, einheitsname_not_found, ncar