import pickle
import os
import pandas as pd

import lightgbm as lgb

import sys
sys.path.append(os.getcwd())

from plot_functions import Visualization
from src.utils import store_trained_model, load_training_data
from src.deployment.inference import Identifier

from evaluation import calculate_metrics, get_best_metric_results
from pathlib import Path

import yaml
from yaml.loader import SafeLoader
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)


model_folder_path = "src/training_pipeline/trained_models/20231127_2048_catboost_HyperparameterTuning/Binary_model"
method = 'catboost'         #lgbm, xgboost, oder catboost
binary_model = True         
index_best_model = 24       #see results (csv files)

with open(os.path.join(model_folder_path, "model_results_dict.pkl"), 'rb') as handle:
    model_results_dict = pickle.load(handle)

#df = pd.read_csv(os.path.join(model_folder_path, "crossvalidation_results.csv"))

#df_cv = df.sort_values(by=["avg validation fbeta_score"], ascending=False)##

#index_best_model = int(df_cv["index"].iloc[0])

print("index_best_model", index_best_model)

best_model = model_results_dict["models_list"][index_best_model]
best_iteration = model_results_dict["best_iteration"][index_best_model]

print("best_iteration", best_iteration)

data_folder = Path(config["train_settings"]["folder_processed_dataset"])
X_train, y_train, X_val, y_val, X_test, y_test, df_preprocessed, df_train, df_val, df_test, weight_factor = load_training_data(data_folder, binary_model=binary_model)


'''
evals = {}
callbacks = [lgb.early_stopping(config["train_settings"]["early_stopping"], verbose=5), lgb.record_evaluation(evals)]

best_model.fit(X_test, y_test,
        eval_set=(X_test, y_test), 
        eval_metric='binary_logloss',
        callbacks=callbacks
        )
'''
evals = model_results_dict["evals_list"][index_best_model]
train_fbeta, train_loss, val_fbeta, val_loss = get_best_metric_results(evals, best_iteration, method, binary_model)

print("train_fbeta: ", train_fbeta)
print("train_loss: ", train_loss)
print("val_fbeta: ", val_fbeta)
print("val_loss: ", val_loss)

y_pred, probs, _  = Identifier.model_predict(model=best_model, X_test=X_val, method=method, binary_model=binary_model)
precision, sensitivity, fbeta, logloss = calculate_metrics(y_val, y_pred, binary_model, probs)

print("Validaion:\n")
print("Precision: ", precision)
print("sensitivity: ", sensitivity)
print("fbeta: ", fbeta)
print("logloss: ", logloss)

y_pred, probs, _  = Identifier.model_predict(model=best_model, X_test=X_test, method=method, binary_model=binary_model)
precision, sensitivity, fbeta, logloss = calculate_metrics(y_test, y_pred, binary_model, probs)

print("\nTest:\n")
print("Precision: ", precision)
print("sensitivity: ", sensitivity)
print("fbeta: ", fbeta)
print("logloss: ", logloss)

'''
if method == 'lgbm':
    Visualization.store_metrics(evals=model_results_dict["evals_list"][index_best_model], best_iteration=best_iteration, model_folder_path=model_folder_path, binary_model=binary_model, finalmodel=False)
else:
    Visualization.plot_metric_custom(evals=model_results_dict["evals_list"][index_best_model], best_iteration=best_iteration, model_folder_path=model_folder_path, method=method, binary=binary_model, finalmodel=False)
'''

#Visualization.store_confusion_matrix(y_test=y_test, y_pred=y_pred, model_folder_path=model_folder_path, binary_model=binary_model)


