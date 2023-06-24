# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
from loguru import logger
import warnings
import os

import lightgbm as lgb
from lightgbm import LGBMClassifier
from lightgbm import plot_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.metrics import ConfusionMatrixDisplay

from Data_Preprocessing import load_prepare_dataset
from train_evaluation_functions import store_predictions, store_trained_model, evaluate_lgbm_model, store_metrics
from config import lgbm_params, general_params, train_settings
from config import lgbm_hyperparameter as lgbm_hp

# %%
warnings.filterwarnings("ignore")

'''
# %%
def store_predictions(model, X_test, y_test, y_pred, probs):
    vectorizer_path = model_folder_path + "vectorizer.pkl"
    # Load the vectorizer from the file
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    vocabulary_path = model_folder_path + "vocabulary.pkl"
    # Get the vocabulary of the training data
    with open(vocabulary_path, 'rb') as f:
        vocabulary = pickle.load(f)
        
    # Extrahieren der wichtigsten Features
    boost = model.booster_
    importance = boost.feature_importance()
    feature_names = boost.feature_name()
    sorted_idx = np.argsort(importance)[::-1]

    feature_dict = {vocabulary.shape[0]+index: key for index, key in enumerate(general_params["features_for_model"])}

    true_label = y_test.reset_index(drop=True)

    X_test_restored = vectorizer.inverse_transform(X_test[:,:vocabulary.shape[0]-len(general_params["features_for_model"])])
    original_designation = [' '.join(words) for words in X_test_restored]

    print('Wichtigsten Features:')
    for j in sorted_idx:
        if importance[j] > 100:
            if j < vocabulary.shape[0]:
                print('{} ({}) Value: {}'.format(feature_names[j], importance[j], vocabulary[j]))
            else:
                print('{} ({}) Value: {}'.format(feature_names[j], importance[j], feature_dict[j]))
        else:
            continue

    # Ausgabe der Vorhersagen, der Wahrscheinlichkeiten und der wichtigsten Features
    for i in range(len(X_test)):
        if y_pred[i] != true_label[i]:
            if y_pred[i] == 1:
                print('Vorhersage für Sample {}: Ja ({})'.format(i+1, y_pred[i]), 'True: Nein ({})'.format(true_label[i]))
            else:
                print('Vorhersage für Sample {}: Nein ({})'.format(i+1, y_pred[i]), 'True: Ja ({})'.format(true_label[i]))
            print(original_designation[i])

            print('Wahrscheinlichkeit für Sample {}: {}'.format(i+1, probs[i][1]))

            print('------------------------')
'''
# %%
def train_model(X_train, y_train, X_val, y_val, weight_factor, lr_index, max_depth_index, feature_frac_index, child_index):
    
    class_weight = {0: 1, 1: weight_factor}
    evals = {}
    callbacks = [lgb.early_stopping(lgbm_params["early_stopping"]), lgb.record_evaluation(evals)]

    gbm = LGBMClassifier(boosting_type=lgbm_params["boosting_type"],
                        objective='binary',
                        metric=lgbm_params["metrics"],
                        num_leaves= pow(2, lgbm_hp["max_depth"][max_depth_index]),
                        max_depth=lgbm_hp["max_depth"][max_depth_index],
                        learning_rate=lgbm_hp['lr'][lr_index],
                        feature_fraction=lgbm_hp["feature_fraction"][feature_frac_index],
                        min_child_samples=lgbm_hp["min_child_samples"][child_index],
                        n_estimators=lgbm_params["n_estimators"],
                        class_weight=class_weight)

    gbm.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)], 
            eval_metric=lgbm_params["metrics"],
            early_stopping_rounds=lgbm_params["early_stopping"],
            callbacks=callbacks)

    
    return gbm, evals

'''
def plot_metrics(model, X_test, y_test, evals, sensitivity):
    probs = model.predict_proba(X_test)
    y_pred = (probs[:,1] >= lgbm_params["prediction_threshold"])
    y_pred =  np.where(y_pred, 1, 0) 

    # Print accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print("\n\n Test accuracy:", accuracy)
    print("Test Sensitivity:", sensitivity, "\n\n")


    lgb.plot_metric(evals, metric=lgbm_params["metrics"][1])
    plt.xlabel('Iterationen')
    plt.ylabel('Loss')
    plt.savefig(model_folder_path + 'binary_logloss_plot.png')

    lgb.plot_metric(evals, metric=lgbm_params["metrics"][0])
    plt.xlabel('Iterationen')
    plt.ylabel('AUC')
    plt.savefig(model_folder_path + 'auc_plot.png')

    class_names = ["Not relevant", "Relevant"]
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, cmap='Blues', colorbar=False)
    plt.savefig(model_folder_path + 'confusion_matrix.png')  

    return accuracy  
'''
# %%
def train_lgbm_binary_model(folder_path):

    global model_folder_path 
    model_folder_path = folder_path + "Binary_model/"
    try:
        os.makedirs(model_folder_path)
    except FileExistsError:
        # directory already exists
        pass

    if train_settings["cross_validation"]==False:
        # Split dataset
        X_train, y_train, X_val, y_val, X_test, y_test, weight_factor = load_prepare_dataset(test_size=lgbm_params["test_size"], folder_path=folder_path, model_folder_path=model_folder_path, binary_model=True)

        df = pd.DataFrame(columns=["model_name","learningrate","max_depth","num_leaves","feature_freaction","min_child_samples","accuracy", "sensitivity"])
        max_sensitivity = 0
        best_lr_index = 0
        best_md_index = 0
        best_ff_index = 0
        best_ci_index = 0
        num_models_trained = 0
        total_models = len(lgbm_hp["lr"]) * len(lgbm_hp["max_depth"]) * len(lgbm_hp["feature_fraction"]) * len(lgbm_hp["min_child_samples"])
        for lr_index in range(len(lgbm_hp["lr"])):
            for max_depth_index in range(len(lgbm_hp["max_depth"])):
                for feature_frac_index in range(len(lgbm_hp["feature_fraction"])):
                    for child_index in range(len(lgbm_hp["min_child_samples"])):
                        gbm, evals = train_model(X_train, y_train, X_val, y_val, weight_factor, lr_index, max_depth_index, feature_frac_index, child_index)
                        y_pred, probs, sensitivity, df_new = evaluate_lgbm_model(gbm, X_test, y_test, lr_index, max_depth_index, feature_frac_index, child_index, num_models_trained, binary_model=True)
                        df = pd.concat([df, df_new])
                        print(f"Modell {df.shape[0]}/{total_models} trained with a sensitivity = {sensitivity}")
                        num_models_trained = num_models_trained + 1
                        if sensitivity > max_sensitivity:
                            best_lr_index = lr_index
                            best_md_index = max_depth_index
                            best_ff_index = feature_frac_index
                            best_ci_index = child_index
                            max_sensitivity = sensitivity
                        break
                    break
                break
            break

        df.to_excel(model_folder_path + "lgbm_hyperparametertuning_results.xlsx")

        gbm, evals = train_model(X_train, y_train, X_val, y_val, weight_factor, best_lr_index, best_md_index, best_ff_index, best_ci_index)
        y_pred, probs, test_acc, sensitivity, df_new = evaluate_lgbm_model(gbm, X_test, y_test, lr_index, max_depth_index, feature_frac_index, child_index, num_models_trained, binary_model=True)
        store_trained_model(gbm, sensitivity, model_folder_path)
        store_predictions(gbm, X_test, y_test, y_pred, probs, folder_path, model_folder_path)
        test_acc = store_metrics(gbm, X_test, y_test, evals, sensitivity, lgbm_params["metrics"], model_folder_path, binary_model=True)
        
        plot_importance(gbm, max_num_features=10)

    else:
        # Split dataset
        X_train, y_train, X_val, y_val, X_test, y_test, weight_factor = load_prepare_dataset(test_size=lgbm_params["test_size"], folder_path=folder_path, model_folder_path=model_folder_path, binary_model=True)

        kfold = KFold(n_splits=train_settings["k-folds"], shuffle=True, random_state=42)
        evals_list = []

        for train_index, val_index in kfold.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            gbm, evals = train_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold, weight_factor)
            evals_list.append(evals)

            y_pred, probs, test_acc, sensitivity, df_new = evaluate_lgbm_model(gbm, X_test, y_test, lr_index, max_depth_index, feature_frac_index, child_index, num_models_trained, binary_model=True)

        store_predictions(gbm, X_test, y_test, y_pred, probs, folder_path, model_folder_path)
        store_trained_model(gbm, sensitivity, model_folder_path)
