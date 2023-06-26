# Used to train the models
# %%
import pandas as pd

import warnings
import os
import time

import lightgbm as lgb
from lightgbm import plot_importance
from sklearn.model_selection import KFold

from models.LightGBM import binary_classifier, multiclass_classifier
from data.preprocessing import load_prepare_dataset
from models.evaluation import store_predictions, store_trained_model, evaluate_lgbm_model, store_metrics
from configs.config_model import lgbm_params_multiclass, train_settings
from configs.config_model import lgbm_hyperparameter as lgbm_hp

warnings.filterwarnings("ignore")

# %%
def model_fit(X_train, y_train, X_val, y_val, weight_factor, lr_index, max_depth_index, feature_frac_index, child_index, binary_model):
    evals = {}
    callbacks = [lgb.early_stopping(train_settings["early_stopping"]), lgb.record_evaluation(evals)]

    if binary_model:
        gbm, metrics = binary_classifier(weight_factor, lr_index, max_depth_index, feature_frac_index, child_index)
    else:
        gbm, metrics = multiclass_classifier(weight_factor, lr_index, max_depth_index, feature_frac_index, child_index)

    gbm.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)], 
            eval_metric=metrics,
            early_stopping_rounds=train_settings["early_stopping"],
            callbacks=callbacks,
            )
    
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

    if train_settings["cross_validation"]==False:
        # Split dataset
        X_train, y_train, X_val, y_val, X_test, y_test, weight_factor = load_prepare_dataset(test_size=lgbm_params_multiclass["test_size"], folder_path=folder_path, model_folder_path=model_folder_path, binary_model=binary_model)
        
        df = pd.DataFrame(columns=["model_name","learningrate","max_depth","num_leaves","feature_freaction","min_child_samples","accuracy", "sensitivity"])
        max_sensitivity = 0
        num_models_trained = 0
        best_model = []
        best_evals = []
        total_models = len(lgbm_hp["lr"]) * len(lgbm_hp["max_depth"]) * len(lgbm_hp["feature_fraction"]) * len(lgbm_hp["min_child_samples"])
        for lr_index in range(len(lgbm_hp["lr"])):
            for max_depth_index in range(len(lgbm_hp["max_depth"])):
                for feature_frac_index in range(len(lgbm_hp["feature_fraction"])):
                    for child_index in range(len(lgbm_hp["min_child_samples"])):
                        start = time.time()
                        gbm, evals = model_fit(X_train, y_train, X_val, y_val, weight_factor, lr_index, max_depth_index, feature_frac_index, child_index, binary_model)
                        stop = time.time()
                        training_time = stop - start
                        y_pred, probs, test_acc, sensitivity, df_new = evaluate_lgbm_model(gbm, X_test, y_test, lr_index, max_depth_index, feature_frac_index, child_index, num_models_trained, training_time, binary_model=binary_model)
                        df = pd.concat([df, df_new])
                        print(f"Modell {df.shape[0]}/{total_models} trained with a sensitivity = {sensitivity}")
                        num_models_trained = num_models_trained + 1
                        if sensitivity > max_sensitivity:
                            best_model.append(gbm)
                            best_evals.append(evals) 
                            max_sensitivity = sensitivity

        df.to_excel(model_folder_path + "lgbm_hyperparametertuning_results.xlsx")

        test_acc = store_metrics(best_model[-1], X_test, y_test, best_evals[-1], sensitivity, model_folder_path, binary_model=binary_model)
        store_trained_model(best_model[-1], test_acc, model_folder_path)

        plot_importance(best_model[-1], max_num_features=10)

        store_predictions(gbm, X_test, y_test, y_pred, probs, folder_path, model_folder_path, binary_model)
    else:
        # Split dataset
        X_train, y_train, X_val, y_val, X_test, y_test, weight_factor = load_prepare_dataset(test_size=lgbm_params_multiclass["test_size"], folder_path=folder_path, model_folder_path=model_folder_path, binary_model=False)

        kfold = KFold(n_splits=train_settings["k-folds"], shuffle=True, random_state=42)
        evals_list = []

        for train_index, val_index in kfold.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            gbm, evals = model_fit(X_train, y_train, X_val, y_val, weight_factor, lr_index, max_depth_index, feature_frac_index, child_index)
            evals_list.append(evals)

            y_pred, probs, test_acc, sensitivity, df_new = evaluate_lgbm_model(gbm, X_test, y_test, lr_index, max_depth_index, feature_frac_index, child_index, num_models_trained, binary_model=False)

        store_predictions(gbm, X_test, y_test, y_pred, probs, folder_path, model_folder_path)
        store_trained_model(best_model[-1], test_acc, model_folder_path)