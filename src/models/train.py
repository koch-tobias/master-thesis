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

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import shap

warnings.filterwarnings("ignore")

# %%
def model_fit(X_train, y_train, X_val, y_val, weight_factor, lr, max_depth, colsample, child, binary_model):
    evals = {}
    callbacks = [lgb.early_stopping(train_settings["early_stopping"], verbose=5), lgb.record_evaluation(evals)]

    if binary_model:
        gbm, metrics = binary_classifier(weight_factor, lr, max_depth, colsample, child)
    else:
        gbm, metrics = multiclass_classifier(weight_factor, lr, max_depth, colsample, child)

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
def train_lgbm_model(folder_path, binary_model):
    global model_folder_path 
    if binary_model:
        model_folder_path = folder_path + "Binary_model/"
    else:
        model_folder_path = folder_path + "Multiclass_model/"

    try:
        os.makedirs(model_folder_path)
    except FileExistsError:
        # directory already exists
        pass

    df = pd.DataFrame(columns=["model_name","learningrate","max_depth","num_leaves","colsample_bytree","min_child_samples", "train auc", "train loss", "validation auc", "validation loss", "test accuracy", "test sensitivity", "Training Time (s)"])
    num_models_trained = 0
    models_list = []
    evals_list = []
    sensitivity_list = []
    y_pred_list = []
    probs_list = []
    total_models = len(lgbm_hp["lr"]) * len(lgbm_hp["max_depth"]) * len(lgbm_hp["colsample_bytree"]) * len(lgbm_hp["min_child_samples"])

    # Training with the hold out method and grid search hyperparameter tuning
    X_train, y_train, X_val, y_val, X_test, y_test, weight_factor = load_prepare_dataset(test_size=train_settings["test_size"], folder_path=folder_path, model_folder_path=model_folder_path, binary_model=binary_model)

    logger.info("Start training with grid search hyperparameter tuning...")
    for lr in lgbm_hp["lr"]:
        for max_depth in lgbm_hp["max_depth"]:
            for colsample in lgbm_hp["colsample_bytree"]:
                for child in lgbm_hp["min_child_samples"]:
                    start = time.time()
                    gbm, evals = model_fit(X_train, y_train, X_val, y_val, weight_factor, lr, max_depth, colsample, child, binary_model)
                    stop = time.time()
                    training_time = stop - start
                    y_pred, probs, test_acc, sensitivity, df_new = evaluate_lgbm_model(gbm, X_test, y_test, evals, lr, max_depth, colsample, child, num_models_trained, training_time, binary_model=binary_model)
                    df = pd.concat([df, df_new])
                    val_acc = evals["valid_1"]["auc"][gbm.best_iteration_-1]
                    val_loss = evals["valid_1"]["binary_logloss"][gbm.best_iteration_-1]
                    print(f"Modell {df.shape[0]}/{total_models} trained with a evaluation accuracy = {val_acc} and with a evaluation loss = {val_loss}")
                    models_list.append(gbm)
                    evals_list.append(evals) 
                    sensitivity_list.append(sensitivity)
                    y_pred_list.append(y_pred)
                    probs_list.append(probs)
                    num_models_trained = num_models_trained + 1
                    break
                break
            break
        break
    logger.success("Grid search hyperparameter tuning was successfull!")

    df.to_excel(model_folder_path + "lgbm_hyperparametertuning_results.xlsx")

    # K-Fold Cross-Validation with the 5 best model trained with the hold out method 
    df = df.sort_values(by=["validation auc"], ascending=False)

    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)

    kfold = StratifiedKFold(n_splits=train_settings["k-folds"], shuffle=True, random_state=general_params["seed"])

    df_cv = pd.DataFrame(columns=["model_name","learningrate","max_depth","num_leaves","colsample_bytree","min_child_samples", "avg train auc", "avg train loss", "avg validation auc", "avg validation loss", "avg test accuracy", "avg test sensitivity"])
    top_x_models = math.ceil(df.shape[0] * 0.10)
    number_of_folds = train_settings["k-folds"]
    total_cv_models = number_of_folds * top_x_models
    count_trained_models = 1

    features, topx_important_features = get_features(models_list[0], model_folder_path)

    logger.info(f"Start to validate the top {top_x_models} models by using {number_of_folds}-fold cross-validation... ")
    for i in range(top_x_models):
        test_sensitivity_results = []
        test_acc_results = []
        val_auc_results = []
        val_loss_results = []
        train_auc_results = []
        train_loss_results = []
        list_shap_values = []
        list_val_sets = []

        for train_index, val_index in kfold.split(X, y):
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            X_train_df = pd.DataFrame(X_train_fold, columns=features)
            X_val_df = pd.DataFrame(X_val_fold, columns=features)           
            
            start = time.time()
            gbm, evals = model_fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold, weight_factor, df["learningrate"].iloc[i], df["max_depth"].iloc[i], df["colsample_bytree"].iloc[i], df["min_child_samples"].iloc[i], binary_model)
            stop = time.time()
            training_time = stop - start
            y_pred, probs, test_acc, test_sensitivity, df_new = evaluate_lgbm_model(gbm, X_test, y_test, evals, df["learningrate"].iloc[i], df["max_depth"].iloc[i], df["colsample_bytree"].iloc[i], df["min_child_samples"].iloc[i], count_trained_models, training_time, binary_model=binary_model)
            val_auc = evals["valid_1"]["auc"][gbm.best_iteration_-1]
            val_loss = evals["valid_1"]["binary_logloss"][gbm.best_iteration_-1]
            train_auc = evals["training"]["auc"][gbm.best_iteration_-1]
            train_loss = evals["training"]["binary_logloss"][gbm.best_iteration_-1]
            logger.info(f"Model {count_trained_models}/{total_cv_models} trained with a evaluation auc = {val_auc} and with a evaluation loss = {val_loss}")
            count_trained_models = count_trained_models + 1
            val_auc_results.append(val_auc)
            val_loss_results.append(val_loss)
            train_auc_results.append(train_auc)
            train_loss_results.append(train_loss)
            test_sensitivity_results.append(test_acc)
            test_acc_results.append(test_sensitivity)
            
            explainer = shap.TreeExplainer(gbm)
            shap_values = explainer.shap_values(X_val_df)
            list_shap_values.append(shap_values)
            list_val_sets.append(val_index)

        avg_val_auc = round(mean(val_auc_results), 6)
        avg_val_loss = round(mean(val_loss_results), 6)
        avg_train_auc = round(mean(train_auc_results), 6)
        avg_train_loss = round(mean(train_loss_results), 6)
        avg_test_sensitivity = round(mean(test_sensitivity_results), 6)
        avg_test_acc = round(mean(test_acc_results), 6)
        df_cv.loc[i, "model_name"] = df["model_name"].iloc[i]
        df_cv.loc[i, "learningrate"] = df["learningrate"].iloc[i]
        df_cv.loc[i, "max_depth"] = df["max_depth"].iloc[i]
        df_cv.loc[i, "num_leaves"] = df["num_leaves"].iloc[i]
        df_cv.loc[i, "colsample_bytree"] = df["colsample_bytree"].iloc[i]
        df_cv.loc[i, "min_child_samples"] = df["min_child_samples"].iloc[i]
        df_cv.loc[i, "avg validation auc"] = avg_val_auc
        df_cv.loc[i, "avg validation loss"] = avg_val_loss      
        df_cv.loc[i, "avg train auc"] = avg_train_auc
        df_cv.loc[i, "avg train loss"] = avg_train_loss     
        df_cv.loc[i, "avg test accuracy"] = avg_test_acc
        df_cv.loc[i, "avg test sensitivity"] = avg_test_sensitivity
        df_cv.loc[i, "index"] = df.index[i]

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
        
    logger.success("Cross-Validation was successfull!")

    df_cv.to_excel(model_folder_path + "lgbm_crossvalidation_results.xlsx")
    df_cv = df_cv.sort_values(by=["avg validation auc"], ascending=False)

    index_best_model = int(df_cv["index"].iloc[0])

    logger.info("Start storing the best model with metrics plots and wrong predictions according to the validation auc...")
    store_metrics(models_list[index_best_model], X_test, y_test, evals_list[index_best_model], model_folder_path, binary_model=binary_model)
    store_trained_model(models_list[index_best_model], df_cv["avg validation auc"].iloc[0], model_folder_path)
    store_predictions(X_test, y_test, y_pred_list[index_best_model], probs_list[index_best_model], folder_path, model_folder_path, binary_model)
    logger.success("Storing was successfull!")

    # Train the best model identified by the holdout method and crossvalidation on the entire data
    logger.info("Train a new model on the entire data with the parameters of the best identified model...")
    X = np.concatenate((X_train, X_val, X_test), axis=0)
    y = np.concatenate((y_train, y_val, y_test), axis=0)
    gbm_final, evals_final = model_fit(X, y, 0, 0, weight_factor, df["learningrate"].iloc[index_best_model], df["max_depth"].iloc[index_best_model], df["colsample_bytree"].iloc[index_best_model], df["min_child_samples"].iloc[index_best_model], binary_model)
    store_trained_model(gbm_final, -1, model_folder_path)
    logger.success("Training the model on the entire data was successfull!")