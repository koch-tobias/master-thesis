# %%
import pandas as pd
import numpy as np

import pickle
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from config_model import general_params, lgbm_params_binary, lgbm_params_multiclass, xgb_params_binary, xgb_params_multiclass, train_settings

# %%
def store_predictions(X_test, y_test, y_pred, probs, folder_path, model_folder_path, binary_model):

    df_test = pd.read_excel(folder_path + "df_testset.xlsx")
    df_preprocessed = pd.read_excel(folder_path + "df_trainset.xlsx") 

    if binary_model:
        class_names = df_preprocessed['Relevant fuer Messung'].unique()
    else:
        class_names = df_preprocessed["Einheitsname"].unique()
        class_names = sorted(class_names)

    df_wrong_predictions = pd.DataFrame(columns=['Benennung (dt)', 'Predicted', 'True', 'Probability'])
    # Ausgabe der Vorhersagen, der Wahrscheinlichkeiten und der wichtigsten Features
    for i in range(len(X_test)):
        if y_pred[i] != y_test[i]:
            df_wrong_predictions.loc[i,"Benennung (dt)"] = df_test.loc[i, "Benennung (dt)"]
            df_wrong_predictions.loc[i,"Predicted"] = class_names[y_pred[i]]
            df_wrong_predictions.loc[i,"True"] = class_names[y_test[i]]
            if binary_model:
                if probs[i][1] >= 0.5:
                    df_wrong_predictions.loc[i,"Probability"] = probs[i][1]
                else:
                    df_wrong_predictions.loc[i,"Probability"] = 1 - probs[i][1]
            else:
                df_wrong_predictions.loc[i,"Probability"] = probs[i][1]
    
    # Serialize data into file:
    df_wrong_predictions.to_excel(model_folder_path + "wrong_predictions.xlsx")

#%%
def get_features(model, model_folder_path, folder_path):
    vocabulary_path = model_folder_path + "vocabulary.pkl"
    # Get the vocabulary of the training data
    with open(vocabulary_path, 'rb') as f:
        vocabulary = pickle.load(f)

    # Extrahieren der wichtigsten Features
    boost = model.booster_
    importance = boost.feature_importance()
    column = boost.feature_name()
    feature_dict = {vocabulary.shape[0]+index: key for index, key in enumerate(general_params["features_for_model"])}

    path_store_features = folder_path + "features.xlsx"
    if os.path.exists(path_store_features):
        df_features = pd.read_excel(path_store_features)
    else:
        df_features = pd.DataFrame(columns=['Column','Feature','Importance Score'])
        for j in range(len(column)):
            df_features.loc[j,"Column"] = column[j]
            df_features.loc[j,"Importance Score"] = importance[j]
            if j < vocabulary.shape[0]:
                df_features.loc[j,"Feature"] = vocabulary[j]
            else:
                df_features.loc[j,"Feature"] = feature_dict[j]

        df_features.to_excel(path_store_features)
    
    topx_important_features = df_features.sort_values(by=["Importance Score"], ascending=False).head(20)
    topx_important_features = topx_important_features.index.tolist()
    features = df_features["Feature"].values.tolist()

    return features, topx_important_features

# %%
def store_trained_model(model, val_auc, model_folder_path):
    # save model
    if val_auc != -1:
        model_path = model_folder_path + f"model_{str(val_auc)[2:6]}_validation_auc.pkl"
    else:
        model_path = model_folder_path + f"final_model.pkl"

    with open(model_path, "wb") as filestore:
        pickle.dump(model, filestore)

# %%
def evaluate_lgbm_model(model, X_test, y_test, evals, hp_in_iteration, num_models_trained, training_time, df_columns, binary_model, method):
    if method == "lgbm":
        best_iteration = model._best_iteration - 1
        probs = model.predict_proba(X_test, num_iteration=best_iteration)
    else:
        best_iteration = model.get_booster().best_ntree_limit - 1
        probs = model.predict_proba(X_test, ntree_limit=best_iteration)

    if binary_model:
        y_pred = (probs[:,1] >= lgbm_params_binary["prediction_threshold"])
        y_pred =  np.where(y_pred, 1, 0)
    else:
        y_pred = probs.argmax(axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, average='weighted')

    df_new = pd.DataFrame(columns=df_columns)

    if method == "lgbm":
        valid_name = "valid_1"
        training_name = "training"
        if binary_model:
            auc = lgbm_params_binary["metrics"][0] 
            loss = lgbm_params_binary["metrics"][1] 
        else:
            auc = lgbm_params_multiclass["metrics"][0] 
            loss = lgbm_params_multiclass["metrics"][1] 
    elif method == "xgboost":
        valid_name = "validation_1"
        training_name = "validation_0"
        if binary_model:
            auc = xgb_params_binary["metrics"][0]
            loss = xgb_params_binary["metrics"][1] 
        else:
            auc = xgb_params_multiclass["metrics"][0]
            loss = xgb_params_multiclass["metrics"][1]

    val_auc = evals[valid_name][auc][best_iteration]
    val_loss = evals[valid_name][loss][best_iteration]
    train_auc = evals[training_name][auc][best_iteration]
    train_loss = evals[training_name][loss][best_iteration]

    df_new.loc[num_models_trained, "model_name"] = f"model_{str(val_auc)[2:6]}"
    for hp in hp_in_iteration:
        df_new.loc[num_models_trained, hp] = hp_in_iteration[hp]
    df_new.loc[num_models_trained, "train auc"] = train_auc
    df_new.loc[num_models_trained, "train loss"] = train_loss
    df_new.loc[num_models_trained, "validation auc"] = val_auc
    df_new.loc[num_models_trained, "validation loss"] = val_loss
    df_new.loc[num_models_trained, "test accuracy"] = accuracy
    df_new.loc[num_models_trained, "test sensitivity"] = sensitivity
    df_new.loc[num_models_trained, "early stopping (iterations)"] = train_settings["early_stopping"]
    df_new.loc[num_models_trained, "Training Time (s)"] = training_time 

    return y_pred, probs, accuracy, sensitivity, val_auc, val_loss, train_auc, train_loss, df_new