# Used to train the models
# %%
import pandas as pd
import numpy as np

import warnings
from loguru import logger
import os
import time
import math
from statistics import mean

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from numba import jit, cuda

from model_architectures import binary_classifier, multiclass_classifier
from evaluation import evaluate_model, add_feature_importance, get_best_metric_results, get_features
from plot_functions import store_metrics, plot_metric_custom, store_confusion_matrix
from utils import store_trained_model, load_dataset
from src.deployment_pipeline.prediction import model_predict, store_predictions, get_best_iteration 
from src.config import train_settings, general_params, paths
from src.config import lgbm_hyperparameter as lgbm_hp
from src.config import xgb_hyperparameter as xgb_hp
from src.config import cb_hyperparameter as cb_hp

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import shap

warnings.filterwarnings("ignore")

try:
    print('System has %d CUDA devices' % len(cuda.list_devices()))
    cuda.select_device(0)
    device = cuda.get_current_device()
    print(device)
except:
    print('System has no CUDA devices')

# %%
def model_fit(X_train, y_train, X_val, y_val, weight_factor, hp_in_iteration, binary_model, method):
    evals = {}
    if method == "lgbm":
        callbacks = [lgb.early_stopping(train_settings["early_stopping"], verbose=5), lgb.record_evaluation(evals)]
    elif method == "xgboost":
        callbacks = [xgb.callback.EarlyStopping(train_settings["early_stopping"], metric_name='auc')]

    if binary_model:
        model, metrics = binary_classifier(weight_factor, hp_in_iteration, method)
    else:
        model, metrics = multiclass_classifier(weight_factor, hp_in_iteration, method)
    
    if method == "lgbm":
        model.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)], 
                eval_metric=metrics,
                callbacks=callbacks
                )

    elif method == "xgboost":
        model.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)], 
                callbacks=callbacks
                )      

        evals = model.evals_result()    
          
    elif method == "catboost":
        model.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)], 
                silent=True
                )

        evals = model.get_evals_result()

    return model, evals

# %%
def get_model_folder_path(binary_model, folder_path):
    if binary_model:
        model_folder_path = folder_path + "Binary_model/"
    else:
        model_folder_path = folder_path + "Multiclass_model/"

    try:
        os.makedirs(model_folder_path)
    except FileExistsError:
        print(f"{model_folder_path}, already exist!")
        # directory already exists
        pass

    return model_folder_path

# %%
def fit_eval_model(X_train, y_train, X_val, y_val, X_test, y_test, weight_factor, hp_in_iteration, df, model_results_dict, num_models_trained, total_models, binary_model, method):
    start = time.time()
    gbm, evals = model_fit(X_train, y_train, X_val, y_val, weight_factor, hp_in_iteration, binary_model, method)
    stop = time.time()
    training_time = int(stop - start)
    y_pred, probs, test_accuracy, test_sensitivity, val_auc, val_loss, train_auc, train_loss, df_new = evaluate_model(gbm, X_test, y_test, evals, hp_in_iteration, num_models_trained, training_time, df.columns, binary_model=binary_model, method=method)    
    df = pd.concat([df, df_new])
    logger.info(f"Modell {num_models_trained+1}/{total_models} trained with a evaluation accuracy = {val_auc} and with a evaluation loss = {val_loss}")
    model_results_dict["models_list"].append(gbm)
    model_results_dict["evals_list"].append(evals)
    model_results_dict["test_sensitivity_results"].append(test_sensitivity)
    model_results_dict["y_pred_list"].append(y_pred)
    model_results_dict["test_accuracy_results"].append(test_accuracy)
    model_results_dict["val_auc_results"].append(val_auc)
    model_results_dict["val_loss_results"].append(val_loss)
    model_results_dict["train_auc_results"].append(train_auc)
    model_results_dict["train_loss_results"].append(train_loss)
    model_results_dict["probs_list"].append(probs)
    num_models_trained = num_models_trained + 1
    return model_results_dict, df

@jit(target_backend='cuda')  
def create_result_df(hp_dict):
    fix_columns = ["model_name", "train auc", "train loss", "validation auc", "validation loss", "test accuracy", "test sensitivity", "Training Time (s)"]

    total_models = 1
    for hp in hp_dict:
        total_models = total_models * len(hp_dict[hp])
    
    columns = fix_columns + list(hp_dict.keys())
    df = pd.DataFrame(columns=columns)

    return df, total_models

# %%
def grid_search(X_train, y_train, X_val, y_val, X_test, y_test, weight_factor, hp_dict, binary_model, method):

    logger.info("Start training with grid search hyperparameter tuning...")

    model_results_dict = {"models_list": [], "evals_list": [], "test_sensitivity_results": [], "test_accuracy_results": [], "val_auc_results": [], "val_loss_results": [], "train_auc_results": [], "train_loss_results": [], "y_pred_list": [], "probs_list": []}
    num_models_trained = 0

    df, total_models = create_result_df(hp_dict)
    hp = list(hp_dict.keys())

    for hp_0 in hp_dict[hp[0]]:
        for hp_1 in hp_dict[hp[1]]:
            for hp_2 in hp_dict[hp[2]]:
                for hp_3 in hp_dict[hp[3]]:
                    hp_in_iteration = {hp[0]: hp_0, hp[1]: hp_1, hp[2]: hp_2, hp[3]: hp_3}
                    model_results_dict, df = fit_eval_model(X_train, y_train, X_val, y_val, X_test, y_test, weight_factor, hp_in_iteration, df, model_results_dict, num_models_trained, total_models, binary_model, method)
                    num_models_trained = num_models_trained + 1

    logger.success("Grid search hyperparameter tuning was successfull!")

    return df, model_results_dict


@jit(target_backend='cuda')  
def get_fold_data(X, y, train_index, val_index):

    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    return X_train_fold, X_val_fold, y_train_fold, y_val_fold
                  
def k_fold_crossvalidation(X, y, X_test, y_test, weight_factor, df, model_results_dict, folder_path, method, binary_model):
    kfold = StratifiedKFold(n_splits=train_settings["k-folds"], shuffle=True, random_state=general_params["seed"])

    fix_columns = ["model_name", "avg train auc", "avg train loss", "avg validation auc", "avg validation loss", "avg test accuracy", "avg test sensitivity"]

    if method == "lgbm":
        hp_columns = list(lgbm_hp.keys())
    elif method == "xgboost":
        hp_columns = list(xgb_hp.keys())
    elif method == "catboost":
        hp_columns = list(cb_hp.keys())

    columns = fix_columns + hp_columns
    df_cv = pd.DataFrame(columns=columns)

    top_x_models = math.ceil(df.shape[0] * train_settings["top_x_models_for_cv"])
    number_of_folds = train_settings["k-folds"]
    total_cv_models = number_of_folds * top_x_models
    count_trained_models = 0

    #logger.info("Store the features with importance score...")
    #df_features = add_feature_importance(model, model_folder_path)
    #feature_list, topx_important_features = get_features(model_results_dict["models_list"][0], model_folder_path, folder_path)
    #logger.success("Storing the features was successfull!")

    logger.info(f"Start to validate the top {top_x_models} models by using {number_of_folds}-fold cross-validation... ")
    for i in range(top_x_models):
        cv_results_dict = {"models_list": [], "evals_list": [], "test_sensitivity_results": [], "test_accuracy_results": [], "val_auc_results": [], "val_loss_results": [], "train_auc_results": [], "train_loss_results": [], "y_pred_list": [], "probs_list": []}
        #list_shap_values = []
        #list_val_sets = []

        logger.info(f"Start {number_of_folds}-fold cross-validation for model {i+1}:")
        for train_index, val_index in kfold.split(X, y):
            
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = get_fold_data(X, y, train_index, val_index)

            #X_train_df = pd.DataFrame(X_train_fold, columns=features)
            #X_val_df = pd.DataFrame(X_val_fold, columns=features)           

            hp_dict = {hp_columns[0]: df[hp_columns[0]].iloc[i], hp_columns[1]: df[hp_columns[1]].iloc[i], hp_columns[2]: df[hp_columns[2]].iloc[i], hp_columns[3]: df[hp_columns[3]].iloc[i]}

            cv_results_dict, df_temp = fit_eval_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test, y_test, weight_factor, hp_dict, df, cv_results_dict, count_trained_models, total_cv_models, binary_model, method)
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
        avg_test_auc = round(mean(cv_results_dict["test_accuracy_results"]), 6)
        for key in list(hp_dict.keys()):
            df_cv.loc[i, key] = df[key].iloc[i]
        df_cv.loc[i, "model_name"] = df["model_name"].iloc[i]
        df_cv.loc[i, "avg validation auc"] = avg_val_auc
        df_cv.loc[i, "avg validation loss"] = avg_val_loss      
        df_cv.loc[i, "avg train auc"] = avg_train_auc
        df_cv.loc[i, "avg train loss"] = avg_train_loss     
        df_cv.loc[i, "avg test accuracy"] = avg_test_auc
        df_cv.loc[i, "avg test sensitivity"] = avg_test_sensitivity
        df_cv.loc[i, "early stopping (iterations)"] = int(train_settings["early_stopping"])
        df_cv.loc[i, "index"] = int(df.index[i])

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

    if method == "lgbm":
        hp_dict = lgbm_hp
    elif method == "xgboost":
        hp_dict = xgb_hp
    elif method == "catboost":
        hp_dict = cb_hp
   
    # Training with the hold out method and grid search hyperparameter tuning
    X_train, y_train, X_val, y_val, X_test, y_test, df_preprocessed, df_test, weight_factor = load_dataset(binary_model=binary_model)

    df, model_results_dict = grid_search(X_train, y_train, X_val, y_val, X_test, y_test, weight_factor, hp_dict, binary_model, method)

    df.to_csv(model_folder_path + "lgbm_hyperparametertuning_results.csv")

    # K-Fold Cross-Validation with the 5 best model trained with the hold out method 
    df = df.sort_values(by=["validation auc"], ascending=False)

    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)

    df_cv = k_fold_crossvalidation(X, y, X_test, y_test, weight_factor, df, model_results_dict, folder_path, method, binary_model)

    df_cv.to_csv(model_folder_path + "lgbm_crossvalidation_results.csv")
    df_cv = df_cv.sort_values(by=["avg validation auc"], ascending=False)

    index_best_model = int(df_cv["index"].iloc[0])

    logger.info("Start storing the best model with metrics plots and wrong predictions according to the validation auc...")
    best_model = model_results_dict["models_list"][index_best_model]

    best_iteration = get_best_iteration(best_model, method)

    if method == 'lgbm':
        store_metrics(model_results_dict["evals_list"][index_best_model], best_iteration, model_folder_path, binary_model, finalmodel=False)
    else:
        plot_metric_custom(model_results_dict["evals_list"][index_best_model], best_iteration, model_folder_path, method, binary_model, finalmodel=False)

    hp_keys = list(hp_dict.keys())
    best_hp = {}
    for key in hp_keys:
        best_hp[key] = df[key].iloc[index_best_model]

    store_trained_model(best_model, best_iteration, df_cv["avg validation auc"].iloc[0], best_hp, index_best_model, model_folder_path, finalmodel=False)
    y_pred, probs, best_iteration = model_predict(best_model, X_test, method, binary_model)
    store_confusion_matrix(y_test, y_pred, folder_path, model_folder_path, binary_model)
    store_predictions(y_test, y_pred, probs, df_preprocessed, df_test, model_folder_path, binary_model)
    logger.success("Storing was successfull!")

    logger.info("Train a new model on the entire data with the parameters of the best identified model...")
    X_train = np.concatenate((X_train, X_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)
        
    gbm_final, evals_final = model_fit(X_train, y_train, X_val, y_val, weight_factor, best_hp, binary_model, method)
    best_iteration = get_best_iteration(gbm_final, method)
    _, _, val_auc, _ = get_best_metric_results(evals_final, best_iteration, method, binary_model)
    store_trained_model(gbm_final, best_iteration, val_auc, best_hp, index_best_model, model_folder_path, finalmodel=True)
    if method == 'lgbm':
        store_metrics(evals_final, best_iteration, model_folder_path, binary_model, finalmodel=True)
        df_feature_importance_final_model = add_feature_importance(gbm_final, model_folder_path=paths["folder_processed_dataset"])
        df_feature_importance_final_model.to_csv(model_folder_path + "final_model_feature_importance.csv")
    else:
        plot_metric_custom(evals_final, best_iteration, model_folder_path, method, binary_model, finalmodel=True)

    logger.success("Training the model on the entire data was successfull!")