# Used to train the models
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
from loguru import logger
import os
import time
import math
from statistics import mean

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

from models.LightGBM import binary_classifier, multiclass_classifier
from data.preprocessing import load_prepare_dataset
from models.evaluation import store_predictions, store_trained_model, evaluate_lgbm_model, get_features
from visualization.plot_functions import store_metrics
from config_model import train_settings, general_params
from config_model import lgbm_hyperparameter as lgbm_hp
from config_model import xgb_hyperparameter as xgb_hp

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import shap

warnings.filterwarnings("ignore")

# %%
def model_fit(X_train, y_train, X_val, y_val, weight_factor, hp_dict, binary_model, method):
    evals = {}
    callbacks = [lgb.early_stopping(train_settings["early_stopping"], verbose=5), lgb.record_evaluation(evals)]

    if binary_model:
        gbm, metrics = binary_classifier(weight_factor, hp_dict, method)
    else:
        gbm, metrics = multiclass_classifier(weight_factor, hp_dict, method)

    try:
        gbm.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)], 
                eval_metric=metrics,
                callbacks=callbacks
                )
    except:
        gbm.fit(X_train, y_train)

    return gbm, evals

# %%
def get_model_folder_path(binary_model, folder_path):
    if binary_model:
        model_folder_path = folder_path + "Binary_model/"
    else:
        model_folder_path = folder_path + "Multiclass_model/"

    try:
        os.makedirs(model_folder_path)
    except FileExistsError:
        # directory already exists
        pass

    return model_folder_path

# %%
def fit_eval_model(X_train, y_train, X_val, y_val, X_test, y_test, weight_factor, hp_dict, df, model_results_dict, num_models_trained, total_models, binary_model, method):
    start = time.time()
    gbm, evals = model_fit(X_train, y_train, X_val, y_val, weight_factor, hp_dict, binary_model, method)
    logger.info("fit successfull")
    stop = time.time()
    training_time = stop - start
    y_pred, probs, test_accuracy, test_sensitivity, val_auc, val_loss, train_auc, train_loss, df_new = evaluate_lgbm_model(gbm, X_test, y_test, evals, hp_dict, num_models_trained, training_time, df.columns, binary_model=binary_model)    
    df = pd.concat([df, df_new])
    logger.info(f"Modell {df.shape[0]}/{total_models} trained with a evaluation accuracy = {val_auc} and with a evaluation loss = {val_loss}")
    model_results_dict["models_list"].append(gbm)
    model_results_dict["evals_list"].append(evals)
    model_results_dict["test_sensitivity_results"].append(test_sensitivity)
    model_results_dict["y_pred_list"].append(y_pred)
    model_results_dict["test_auc_results"].append(test_accuracy)
    model_results_dict["val_auc_results"].append(val_auc)
    model_results_dict["val_loss_results"].append(val_loss)
    model_results_dict["train_auc_results"].append(train_auc)
    model_results_dict["train_loss_results"].append(train_loss)
    model_results_dict["probs_list"].append(probs)
    num_models_trained = num_models_trained + 1
    return model_results_dict, df

# %%
def grid_search(X_train, y_train, X_val, y_val, X_test, y_test, weight_factor, binary_model, method):

    logger.info("Start training with grid search hyperparameter tuning...")

    num_models_trained = 0
    total_models = 1
    for hp in lgbm_hp:
        total_models = total_models * len(lgbm_hp[hp])

    model_results_dict = {"models_list": [], "evals_list": [], "test_sensitivity_results": [], "test_auc_results": [], "val_auc_results": [], "val_loss_results": [], "train_auc_results": [], "train_loss_results": [], "y_pred_list": [], "probs_list": []}
    fix_columns = ["model_name", "train auc", "train loss", "validation auc", "validation loss", "test accuracy", "test sensitivity", "Training Time (s)"]
   
    if method=="lgbm":
        hp_columns = list(lgbm_hp.keys())
        columns = fix_columns + hp_columns
        df = pd.DataFrame(columns=columns)

        for lr in lgbm_hp["lr"]:
            for max_depth in lgbm_hp["max_depth"]:
                for colsample in lgbm_hp["colsample_bytree"]:
                    for child in lgbm_hp["min_child_samples"]:
                        hp_dict = {"lr": lr, "max_depth": max_depth, "colsample_bytree": colsample, "min_child_samples": child}
                        model_results_dict, df = fit_eval_model(X_train, y_train, X_val, y_val, X_test, y_test, weight_factor, hp_dict, df, model_results_dict, num_models_trained, total_models, binary_model, method)
    
    elif method=="xgboost":
        hp_columns = list(xgb_hp.keys())
        columns = fix_columns + hp_columns
        df = pd.DataFrame(columns=columns)

        for lr in xgb_hp["lr"]:
            for max_depth in xgb_hp["max_depth"]:
                for colsample in xgb_hp["colsample_bytree"]:
                    for gamma in xgb_hp["gamma"]:
                        hp_dict = {"lr": lr, "max_depth": max_depth, "colsample_bytree": colsample, "gamma": gamma}
                        model_results_dict, df = fit_eval_model(X_train, y_train, X_val, y_val, X_test, y_test, weight_factor, hp_dict, df, model_results_dict, num_models_trained, total_models, binary_model, method)

    logger.success("Grid search hyperparameter tuning was successfull!")

    return df, model_results_dict

def k_fold_crossvalidation(X, y, X_test, y_test, weight_factor, df, model_results_dict, folder_path, method, binary_model):
    kfold = StratifiedKFold(n_splits=train_settings["k-folds"], shuffle=True, random_state=general_params["seed"])

    fix_columns = ["model_name", "avg train auc", "avg train loss", "avg validation auc", "avg validation loss", "avg test accuracy", "avg test sensitivity"]

    if method == "lgbm":
        hp_columns = list(lgbm_hp.keys())
    elif method == "xgboost":
        hp_columns = list(xgb_hp.keys())

    columns = fix_columns + hp_columns
    df_cv = pd.DataFrame(columns=columns)

    top_x_models = math.ceil(df.shape[0] * 0.10)
    number_of_folds = train_settings["k-folds"]
    total_cv_models = number_of_folds * top_x_models
    count_trained_models = 1

    logger.info("Store the features with importance score...")
    features, topx_important_features = get_features(model_results_dict["models_list"][0], model_folder_path, folder_path)
    logger.success("Storing the features was successfull!")

    logger.info(f"Start to validate the top {top_x_models} models by using {number_of_folds}-fold cross-validation... ")
    for i in range(top_x_models):
        cv_results_dict = {"models_list": [], "evals_list": [], "test_sensitivity_results": [], "test_auc_results": [], "val_auc_results": [], "val_loss_results": [], "train_auc_results": [], "train_loss_results": [], "y_pred_list": [], "probs_list": []}
        list_shap_values = []
        list_val_sets = []

        logger.info(f"Start {number_of_folds}-fold cross-validation for model {i+1}:")
        for train_index, val_index in kfold.split(X, y):
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            X_train_df = pd.DataFrame(X_train_fold, columns=features)
            X_val_df = pd.DataFrame(X_val_fold, columns=features)           

            if method == "lgbm":
                hp_dict = {"lr": df["learningrate"].iloc[i], "max_depth": df["max_depth"].iloc[i], "colsample_bytree": df["colsample_bytree"].iloc[i], "min_child_samples": df["min_child_samples"].iloc[i]}
            elif method == "xgboost":
                hp_dict = {"lr": df["learningrate"].iloc[i], "max_depth": df["max_depth"].iloc[i], "colsample_bytree": df["colsample_bytree"].iloc[i], "gamma": df["gamma"].iloc[i]}

            cv_results_dict, df_temp = fit_eval_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test, y_test, weight_factor, hp_dict, df, cv_results_dict, count_trained_models, total_cv_models, binary_model, method)
            val_auc = cv_results_dict["val_auc_results"][-1]
            val_loss = cv_results_dict["val_loss_results"][-1]
            logger.info(f"Model {count_trained_models}/{total_cv_models} trained with a evaluation auc = {val_auc} and with a evaluation loss = {val_loss}")
            count_trained_models = count_trained_models + 1

            '''
            if binary_model:
                explainer = shap.TreeExplainer(gbm)
                shap_values = explainer.shap_values(X_val_df)
                list_shap_values.append(shap_values)
                list_val_sets.append(val_index)
            '''
        avg_val_auc = round(mean(cv_results_dict["val_auc_results"]), 6)
        avg_val_loss = round(mean(cv_results_dict["val_loss_results"]), 6)
        avg_train_auc = round(mean(cv_results_dict["train_auc_results"]), 6)
        avg_train_loss = round(mean(cv_results_dict["train_loss_results"]), 6)
        avg_test_sensitivity = round(mean(cv_results_dict["test_sensitivity_results"]), 6)
        avg_test_auc = round(mean(cv_results_dict["test_auc_results"]), 6)
        for key in list(hp_dict.keys()):
            df_cv.loc[i, key] = df[key].iloc[i]
        df_cv.loc[i, "model_name"] = df["model_name"].iloc[i]
        df_cv.loc[i, "avg validation auc"] = avg_val_auc
        df_cv.loc[i, "avg validation loss"] = avg_val_loss      
        df_cv.loc[i, "avg train auc"] = avg_train_auc
        df_cv.loc[i, "avg train loss"] = avg_train_loss     
        df_cv.loc[i, "avg test accuracy"] = avg_test_auc
        df_cv.loc[i, "avg test sensitivity"] = avg_test_sensitivity
        df_cv.loc[i, "early stopping (iterations)"] = train_settings["early_stopping"]
        df_cv.loc[i, "index"] = df.index[i]

        '''
        if binary_model:
            #combining results from all iterations
            val_set = list_val_sets[0]
            shap_values = np.array(list_shap_values[0])[:, :, topx_important_features]
            
            for i in range(0,len(list_val_sets)):
                val_set = np.concatenate((val_set, list_val_sets[i]),axis=0)
                shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])[:, :, topx_important_features]),axis=1)
        
            #bringing back variable names    
            X_val_df = pd.DataFrame(X[val_set], columns=features)
            #creating explanation plot for the whole experiment
            plt.clf()
            shap.summary_plot(shap_values[1], X_val_df.iloc[:, topx_important_features], show=False)
            plt.savefig(model_folder_path + "shap_top10_features.png")
    '''
    logger.success("Cross-Validation was successfull!")

    return df_cv
# %%
def train_model(folder_path, binary_model, method):
    global model_folder_path 
    model_folder_path = get_model_folder_path(binary_model, folder_path)
   
    # Training with the hold out method and grid search hyperparameter tuning
    X_train, y_train, X_val, y_val, X_test, y_test, weight_factor = load_prepare_dataset(test_size=train_settings["test_size"], folder_path=folder_path, model_folder_path=model_folder_path, binary_model=binary_model)

    df, model_results_dict = grid_search(X_train, y_train, X_val, y_val, X_test, y_test, weight_factor, binary_model, method)

    df.to_excel(model_folder_path + "lgbm_hyperparametertuning_results.xlsx")

    # K-Fold Cross-Validation with the 5 best model trained with the hold out method 
    df = df.sort_values(by=["validation auc"], ascending=False)

    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)

    df_cv = k_fold_crossvalidation(X, y, X_test, y_test, weight_factor, df, model_results_dict, folder_path, method, binary_model)

    df_cv.to_excel(model_folder_path + "lgbm_crossvalidation_results.xlsx")
    df_cv = df_cv.sort_values(by=["avg validation auc"], ascending=False)

    index_best_model = int(df_cv["index"].iloc[0])

    logger.info("Start storing the best model with metrics plots and wrong predictions according to the validation auc...")
    store_metrics(model_results_dict["models_list"][index_best_model], X_test, y_test, model_results_dict["evals_list"][index_best_model], model_folder_path, binary_model=binary_model)
    store_trained_model(model_results_dict["models_list"][index_best_model], df_cv["avg validation auc"].iloc[0], model_folder_path)
    store_predictions(X_test, y_test, model_results_dict["y_pred_list"][index_best_model], model_results_dict["probs_list"][index_best_model], folder_path, model_folder_path, binary_model)
    logger.success("Storing was successfull!")

    # Train the best model identified by the holdout method and crossvalidation on the entire data
    logger.info("Train a new model on the entire data with the parameters of the best identified model...")
    X = np.concatenate((X_train, X_val, X_test), axis=0)
    y = np.concatenate((y_train, y_val, y_test), axis=0)
    hp_dict = {"lr": df["learningrate"].iloc[index_best_model], "max_depth": df["max_depth"].iloc[index_best_model], "colsample_bytree": df["colsample_bytree"].iloc[index_best_model], "min_child_samples": df["min_child_samples"].iloc[index_best_model]}
    gbm_final, evals_final = model_fit(X, y, 0, 0, weight_factor, hp_dict, binary_model, method)
    store_trained_model(gbm_final, -1, model_folder_path)
    logger.success("Training the model on the entire data was successfull!")