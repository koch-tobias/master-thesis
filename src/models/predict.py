import pandas as pd
import numpy as np
import pickle
from loguru import logger
from data.preprocessing import prepare_and_add_labels, preprocess_dataset, get_model, get_X
from data.boundingbox_calculations import find_valid_space
from config_model import paths, lgbm_params_binary

# %%
def model_predict(model, X_test, binary_model):
    probs = model.predict_proba(X_test, num_iteration=model._best_iteration)
    if binary_model:
        y_pred = (probs[:,1] >= lgbm_params_binary["prediction_threshold"])
        y_pred =  np.where(y_pred, 1, 0)
    else:
        y_pred = probs.argmax(axis=1)   
    
    return y_pred, probs

def predict_on_new_data(df):
    logger.info("Prepare dataset...")
    df, ncar = prepare_and_add_labels(df)
    logger.info("Dataset successfully prepared!")

    logger.info("Load trainset..")
    trainset = pd.read_excel("models/df_trainset.xlsx")
    trainset_relevant_parts = trainset[trainset["Relevant fuer Messung"] == "Ja"]
    trainset_relevant_parts = trainset_relevant_parts[(trainset_relevant_parts['X-Min_transf'] != 0) & (trainset_relevant_parts['X-Max_transf'] != 0)]    
    unique_names = trainset_relevant_parts["Einheitsname"].unique().tolist()
    unique_names.sort()
    logger.success("Trainset loaded!")
    
    logger.info("Load pretrained models...")
    lgbm_binary, vectorizer_binary, vocabulary_binary, boundingbox_features_binary = get_model(paths["model_folder"] + '/Binary_model')
    lgbm_multiclass, vectorizer_multiclass, vocabulary_multiclass, boundingbox_features_multiclass = get_model(paths["model_folder"] + '/Multiclass_model')       
    logger.success("Pretrained models loaded!")

    logger.info("Preprocess data...")
    df_preprocessed, df_for_plot = preprocess_dataset(df)

    X_binary = get_X(vocabulary_binary, vectorizer_binary, boundingbox_features_binary["features_for_model"], df_preprocessed)
    y_pred_binary, probs = model_predict(lgbm_binary, X_binary, binary_model=True)

    X_multiclass = get_X(vocabulary_multiclass, vectorizer_multiclass, boundingbox_features_multiclass["features_for_model"], df_preprocessed)
    y_pred_multiclass, probs = model_predict(lgbm_multiclass, X_multiclass, binary_model=False)
    logger.success("Dataset is ready for classification!")

    logger.info("Load LabelEncoder...")
    # Load the LabelEncoder
    with open(paths["model_folder"] + '/Multiclass_model/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f) 
    logger.success("LabelEncoder loaded!")

    y_pred_multiclass_names = le.inverse_transform(y_pred_multiclass) 

    logger.info("Identify relevant car parts and add the unique names...")
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
    logger.success("Relevant car parts identified!")
    
    logger.info("Valid identified car party comparing bounding-box informations to the trainset")
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
    logger.success("Comparing successfull!")

    for name in unique_names:        
        if name not in df_relevant_parts['Einheitsname'].unique():
            einheitsname_not_found.append(name)

    df_relevant_parts = df_relevant_parts.reset_index(drop=True)
    df_relevant_parts["L/R-Kz."] = df_relevant_parts["L/R-Kz."].fillna(' ')
    df_relevant_parts.loc[df_relevant_parts['Einheitsname'] == "Dummy", 'Einheitsname'] = 'Kein Einheitsname gefunden'
    df_relevant_parts.loc[df_relevant_parts["L/R-Kz."] == "L", "L/R-Kz."] = 'Linke Ausfuehrung'

    return df_preprocessed, df_relevant_parts, einheitsname_not_found, ncar
