import pandas as pd
import numpy as np
import pickle
from loguru import logger
from src.data_pipeline.preprocessing import load_data_into_df, prepare_and_add_labels, preprocess_dataset
from src.data_pipeline.feature_engineering import find_valid_space
from src.training_pipeline.utils import get_model, get_X, get_dataset_path_from_logging
from src.config import general_params, paths, prediction_settings

def get_best_iteration(model, method):
    if method == "lgbm":
        best_iteration = model._best_iteration - 1
    elif method == "xgboost":
        best_iteration = model.get_booster().best_ntree_limit - 1
    elif method == "catboost":
        best_iteration = model.get_best_iteration() - 1

    return best_iteration

def get_probabilities(model, X_test, method):
    if method == "lgbm":
        best_iteration = model._best_iteration - 1
        probs = model.predict_proba(X_test, num_iteration=best_iteration)
    elif method == "xgboost":
        best_iteration = model.get_booster().best_ntree_limit - 1
        probs = model.predict_proba(X_test, ntree_limit=best_iteration)
    elif method == "catboost":
        best_iteration = model.get_best_iteration() - 1
        probs = model.predict_proba(X_test)

    return probs
# %%
def model_predict(model, X_test, method, binary_model):
    best_iteration = get_best_iteration(model, method)
    probs = get_probabilities(model, X_test, method)

    if binary_model:
        y_pred = (probs[:,1] >= prediction_settings["prediction_threshold"])
        y_pred =  np.where(y_pred, 1, 0)
    else:
        y_pred = probs.argmax(axis=1)   
    
    return y_pred, probs, best_iteration

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
        if y_pred[i] != y_test[i]:
            df_wrong_predictions.loc[i,"Sachnummer"] = df_test.loc[i, "Sachnummer"]
            df_wrong_predictions.loc[i,"Benennung (dt)"] = df_test.loc[i, "Benennung (dt)"]
            df_wrong_predictions.loc[i,"Derivat"] = df_test.loc[i, "Derivat"]
            df_wrong_predictions.loc[i,"Predicted"] = class_names[y_pred[i]]
            df_wrong_predictions.loc[i,"True"] = class_names[y_test[i]]
            if binary_model:
                if probs[i][1] >= prediction_settings["prediction_threshold"]:
                    df_wrong_predictions.loc[i,"Probability"] = probs[i][1]
                else:
                    df_wrong_predictions.loc[i,"Probability"] = 1 - probs[i][1]
            else:
                df_wrong_predictions.loc[i,"Probability"] = probs[i][1]

        
    # Serialize data into file:
    df_wrong_predictions.to_csv(model_folder_path + "wrong_predictions.csv")

def get_method_from_logging(model_folder_path):
    logging_path = model_folder_path + "/logging.txt"
    with open(logging_path, 'r') as file:
        for line in file:
            if "Method:" in line:
                return line.split(":")[1].strip()
    return None

def predict_on_new_data(df):
    logger.info("Prepare dataset...")
    df, ncar = prepare_and_add_labels(df)
    logger.info("Dataset successfully prepared!")
    
    logger.info("Load pretrained models...")
    model_folder_path_binary = paths["final_model"] + "/Binary_model"
    model_folder_path_multiclass = paths["final_model"] + "/Multiclass_model"
    binary_model, vectorizer_binary, vocabulary_binary, boundingbox_features_binary = get_model(model_folder_path_binary)
    multiclass_model, vectorizer_multiclass, vocabulary_multiclass, boundingbox_features_multiclass = get_model(model_folder_path_multiclass)       
    logger.success("Pretrained models are loaded!")

    logger.info("Preprocess data...")
    df_preprocessed, df_for_plot = preprocess_dataset(df)

    X_binary = get_X(vocabulary_binary, vectorizer_binary, boundingbox_features_binary["features_for_model"], df_preprocessed)
    X_multiclass = get_X(vocabulary_multiclass, vectorizer_multiclass, boundingbox_features_multiclass["features_for_model"], df_preprocessed)
    logger.success("Dataset is ready for classification!")   

    logger.info("Identify relevant car parts and add the unique names...")
    binary_method = get_method_from_logging(model_folder_path_binary)
    multiclass_method = get_method_from_logging(model_folder_path_multiclass)
    y_pred_binary, probs_binary, _  = model_predict(binary_model, X_binary, binary_method, binary_model=True)
    y_pred_multiclass, probs_multiclass, _ = model_predict(multiclass_model, X_multiclass, multiclass_method, binary_model=False)

    dataset_path_binary = get_dataset_path_from_logging(model_folder_path_binary + "/logging.txt")
    dataset_path_multiclass = get_dataset_path_from_logging(model_folder_path_multiclass + "/logging.txt")

    # Load the LabelEncoder
    with open(dataset_path_multiclass + 'label_encoder.pkl', 'rb') as f:
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
    logger.success("Relevant car parts identified!")
    
    logger.info("Valid identified car parts by comparing the position of the new car part to them in the trainset")

    logger.info("Load dataset used for training..")
    trainset = pd.read_csv(dataset_path_binary + "processed_dataset.csv")
    trainset_relevant_parts = trainset[trainset["Relevant fuer Messung"] == "Ja"]
    trainset_relevant_parts = trainset_relevant_parts[(trainset_relevant_parts['X-Min_transf'] != 0) & (trainset_relevant_parts['X-Max_transf'] != 0)]    
    unique_names = trainset_relevant_parts["Einheitsname"].unique().tolist()
    unique_names.sort()
    logger.success("Dataset loaded!")

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
    logger.success("Validation successfull!")

    logger.info("Prepare output...")
    for name in unique_names:        
        if name not in df_relevant_parts['Einheitsname'].unique():
            einheitsname_not_found.append(name)

    df_relevant_parts = df_relevant_parts.reset_index(drop=True)
    df_relevant_parts.loc[df_relevant_parts['Einheitsname'] == "Dummy", 'Einheitsname'] = 'Kein Einheitsname gefunden'

    df_relevant_parts = df_relevant_parts.drop_duplicates(subset=["Sachnummer"])

    logger.success("Output is prepared!")

    return df_preprocessed, df_relevant_parts, einheitsname_not_found, ncar

def label_data():
    dataframes, ncars = load_data_into_df(original_prisma_data=True, label_new_data=True)
    for df in dataframes:
        df_with_label_columns, df_relevant_parts, einheitsname_not_found, ncar = predict_on_new_data(df)

        for index, row in df_relevant_parts.iterrows():
            sachnummer = row['Sachnummer']
            einheitsname = row['Einheitsname']
            
            if sachnummer in df['Sachnummer'].values:
                df_with_label_columns.loc[df_with_label_columns['Sachnummer'] == sachnummer, 'Relevant fuer Messung'] = "Ja"
                df_with_label_columns.loc[df_with_label_columns['Sachnummer'] == sachnummer, 'Einheitsname'] = einheitsname

        features = general_params["relevant_features"] + ['Relevant fuer Messung','Einheitsname']
        df_with_label_columns = df_with_label_columns[features]
        df_with_label_columns.to_csv(f"data/pre_labeled/{ncar}_labeled.csv")

        logger.info(f"The following car parts are not found in your dataset: {einheitsname_not_found} If essential, please add this car parts manually!")
        logger.success(f"The prediction is done and the result is stored here: data/pre_labeled_data/{ncar}_labeled.csv!")

# %%
def main():
    
    label_new_data = False
    make_prediction = True
    data_path = 'data/raw_for_labeling/G60_G60_G60_G60_G60_G60_G60_G60_G60_G60_G60_G60_G60_G60_G60_EP_prismaexport-20230731-171755.xls'

    if make_prediction:
        df = pd.read_excel(data_path, header=None, skiprows=1)
        df.columns = df.iloc[0]
        df = df.iloc[1:] 
        df_preprocessed, df_relevant_parts, einheitsname_not_found, ncar = predict_on_new_data(df)
        print(df_relevant_parts)

    if label_new_data:
        label_data()

# %%
if __name__ == "__main__":
    
    main()