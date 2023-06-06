# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
from loguru import logger
import warnings

import lightgbm as lgb
from lightgbm import LGBMClassifier
from lightgbm import plot_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.metrics import ConfusionMatrixDisplay

from Prepare_data import load_prepare_dataset
from config import lgbm_params, general_params, train_settings
from config import lgbm_hyperparameter as lgbm_hp

# %%
warnings.filterwarnings("ignore")

# %%
def store_predictions(model, X_test, y_test, y_pred, probs, timestamp):
    vectorizer_path = f"../models/lgbm_{timestamp}/vectorizer.pkl"
    # Load the vectorizer from the file
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    vocabulary_path = f"../models/lgbm_{timestamp}/vocabulary.pkl"
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

# %%
def sensitivity_metric(y_true, y_pred):
    y_pred_thresh = [1 if pred >= 0.5 else 0 for pred in y_pred]
    eval_result = recall_score(y_true, y_pred_thresh)
    return eval_result

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

# %%
def store_trained_model(model, test_acc, timestamp):
    # save model
    model_path = f"../models/lgbm_{timestamp}/model_{str(test_acc)[2:]}.pkl"
    with open(model_path, "wb") as filestore:
        pickle.dump(model, filestore)

# %%
def evaluate_model(model, X_test, y_test, timestamp, lr_index, max_depth_index, feature_frac_index, child_index, num_models_trained):
    probs = model.predict_proba(X_test)
    y_pred = (probs[:,1] >= lgbm_params["prediction_threshold"])
    y_pred =  np.where(y_pred, 1, 0) 

    # Print accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = sensitivity_metric(y_test, y_pred)

    df_new = pd.DataFrame(columns=["model_name","learningrate","max_depth","num_leaves","feature_freaction","min_child_samples","accuracy", "sensitivity"])

    df_new.loc[num_models_trained, "model_name"] = "lightgbm_" + str(timestamp)
    df_new.loc[num_models_trained, "learningrate"] = lgbm_hp["lr"][lr_index]
    df_new.loc[num_models_trained, "max_depth"] = lgbm_hp["max_depth"][max_depth_index]
    df_new.loc[num_models_trained, "num_leaves"] = pow(2, lgbm_hp["max_depth"][max_depth_index])
    df_new.loc[num_models_trained, "feature_freaction"] = lgbm_hp["feature_fraction"][feature_frac_index]
    df_new.loc[num_models_trained, "min_child_samples"] = lgbm_hp["min_child_samples"][child_index]
    df_new.loc[num_models_trained, "accuracy"] = accuracy
    df_new.loc[num_models_trained, "sensitivity"] = sensitivity

    return y_pred, probs, accuracy, sensitivity, df_new

def plot_metrics(model, X_test, y_test, evals, sensitivity, timestamp):
    probs = model.predict_proba(X_test)
    y_pred = (probs[:,1] >= lgbm_params["prediction_threshold"])
    y_pred =  np.where(y_pred, 1, 0) 

    # Print accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print("\n\n Test accuracy:", accuracy)
    print("Test Sensitivity:", sensitivity, "\n\n")


    lgb.plot_metric(evals, metric=lgbm_params["metrics"][1])
    plt.savefig(f'../models/lgbm_{timestamp}/binary_logloss_plot.png')

    lgb.plot_metric(evals, metric=lgbm_params["metrics"][0])
    plt.savefig(f'../models/lgbm_{timestamp}/auc_plot.png')

    class_names = ["Nein", "Ja"]
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, cmap='Blues', colorbar=False)
    plt.savefig(f'../models/lgbm_{timestamp}/confusion_matrix.png')  

    return accuracy  

# %%
def main():
    if train_settings["cross_validation"]==False:
        # Split dataset
        X_train, y_train, X_val, y_val, X_test, y_test, weight_factor, timestamp, vocab = load_prepare_dataset(test_size=lgbm_params["test_size"])
        
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
                        y_pred, probs, test_acc, sensitivity, df_new = evaluate_model(gbm, X_test, y_test, timestamp, lr_index, max_depth_index, feature_frac_index, child_index, num_models_trained)
                        df = pd.concat([df, df_new])
                        print(f"Modell {df.shape[0]}/{total_models} trained with a sensitivity = {sensitivity}")
                        num_models_trained = num_models_trained + 1
                        if sensitivity > max_sensitivity:
                            best_lr_index = lr_index
                            best_md_index = max_depth_index
                            best_ff_index = feature_frac_index
                            best_ci_index = child_index

        df.to_excel(f"../models/lgbm_hyperparametertuning_{timestamp}.xlsx")
                       
        if train_settings["print_predictions"]:
            store_predictions(gbm, X_test, y_test, y_pred, probs, timestamp)

        if train_settings["store_trained_model"]:
            gbm, evals = train_model(X_train, y_train, X_val, y_val, weight_factor, best_lr_index, best_md_index, best_ff_index, best_ci_index)
            test_acc = plot_metrics(gbm, X_test, y_test, evals, sensitivity, timestamp)
            store_trained_model(gbm, test_acc, timestamp)

            plot_importance(gbm, max_num_features=10)
    else:
        # Split dataset
        X_train, y_train, X_test, y_test, weight_factor, timestamp, vocab = load_prepare_dataset(test_size=lgbm_params["test_size"])

        kfold = KFold(n_splits=train_settings["k-folds"], shuffle=True, random_state=42)
        evals_list = []

        for train_index, val_index in kfold.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            gbm, evals = train_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold, weight_factor)
            evals_list.append(evals)

            y_pred, test_acc = evaluate_model(gbm, X_test, y_test, evals, timestamp)

        if train_settings["print_predictions"]:
            store_predictions(gbm, X_test, y_test, y_pred, timestamp)

        if train_settings["store_trained_model"]:
            store_trained_model(gbm, test_acc, timestamp) 


# %%
if __name__ == "__main__":
    
    main()
# %%
