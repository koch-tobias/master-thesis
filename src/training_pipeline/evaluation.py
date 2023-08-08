# %%
import pandas as pd

import pickle

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from src.deployment_pipeline.prediction import model_predict

from src.config import general_params, lgbm_params_binary, lgbm_params_multiclass, xgb_params_binary, xgb_params_multiclass, cb_params_binary, cb_params_multiclass, train_settings

#%%
def add_feature_importance(model, model_folder_path):
    vocabulary_path = model_folder_path + "vocabulary.pkl"
    # Get the vocabulary of the training data
    with open(vocabulary_path, 'rb') as f:
        vocabulary = pickle.load(f)

    # Extrahieren der wichtigsten Features
    boost = model.booster_
    importance = boost.feature_importance()
    column = boost.feature_name()
    feature_dict = {vocabulary.shape[0]+index: key for index, key in enumerate(general_params["features_for_model"])}

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
def get_features(model_folder_path):
    path_store_features = model_folder_path + "features.xlsx"
    try:
        df_features = pd.read_excel(path_store_features) 
        topx_important_features = df_features.sort_values(by=["Importance Score"], ascending=False).head(20)
        topx_important_features = topx_important_features.index.tolist()
        feature_list = df_features["Feature"].values.tolist()

        return feature_list, topx_important_features
    except:
        print(f"Error: File {path_store_features} does not exist!")
    
def get_best_metric_results(evals, best_iteration, method, binary_model):
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

    elif method == "catboost":
        valid_name = "validation_1"
        training_name = "validation_0"
        if binary_model:
            auc = cb_params_binary["metrics"][0]
            loss = cb_params_binary["metrics"][1] 
        else:
            if cb_params_multiclass["metrics"][0] == 'AUC':
                auc = 'AUC:type=Mu'
            else:
                auc = cb_params_multiclass["metrics"][0]
            loss = cb_params_multiclass["metrics"][1]        

    val_auc = evals[valid_name][auc][best_iteration]
    val_loss = evals[valid_name][loss][best_iteration]
    train_auc = evals[training_name][auc][best_iteration]
    train_loss = evals[training_name][loss][best_iteration]
    return train_auc, train_loss, val_auc, val_loss

# %%
def evaluate_model(model, X_test, y_test, evals, hp_in_iteration, num_models_trained, training_time, df_columns, binary_model, method):
    
    y_pred, probs, best_iteration  = model_predict(model, X_test, method, binary_model)

    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, average='macro')

    df_new = pd.DataFrame(columns=df_columns)

    train_auc, train_loss, val_auc, val_loss = get_best_metric_results(evals, best_iteration, method, binary_model)

    df_new.loc[num_models_trained, "model_name"] = f"model_{str(val_auc)[2:6]}"
    for hp in hp_in_iteration:
        df_new.loc[num_models_trained, hp] = hp_in_iteration[hp]
    df_new.loc[num_models_trained, "train auc"] = train_auc
    df_new.loc[num_models_trained, "train loss"] = train_loss
    df_new.loc[num_models_trained, "validation auc"] = val_auc
    df_new.loc[num_models_trained, "validation loss"] = val_loss
    df_new.loc[num_models_trained, "test accuracy"] = accuracy
    df_new.loc[num_models_trained, "test sensitivity"] = sensitivity
    df_new.loc[num_models_trained, "early stopping (iterations)"] = int(train_settings["early_stopping"])
    df_new.loc[num_models_trained, "Training Time (s)"] = training_time 

    return y_pred, probs, accuracy, sensitivity, val_auc, val_loss, train_auc, train_loss, df_new