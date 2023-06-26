# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
import os

import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import ConfusionMatrixDisplay

from config import general_params, lgbm_params_binary, lgbm_params_multiclass
from config import lgbm_hyperparameter as lgbm_hp

# %%
def store_predictions(model, X_test, y_test, y_pred, probs, folder_path, model_folder_path, binary_model):

    vocabulary_path = model_folder_path + "vocabulary.pkl"
    # Get the vocabulary of the training data
    with open(vocabulary_path, 'rb') as f:
        vocabulary = pickle.load(f)

    df_test = pd.read_excel(folder_path + "df_testset.xlsx")
    df_preprocessed = pd.read_excel(folder_path + "df_trainset.xlsx") 

    if binary_model:
        class_names = df_preprocessed['Relevant fuer Messung'].unique()
    else:
        class_names = df_preprocessed["Einheitsname"].unique()
        class_names = sorted(class_names)

    # Extrahieren der wichtigsten Features
    boost = model.booster_
    importance = boost.feature_importance()
    column = boost.feature_name()
    feature_dict = {vocabulary.shape[0]+index: key for index, key in enumerate(general_params["features_for_model"])}

    if os.path.exists("models/features.xlsx"):
        pass
    else:
        df_features = pd.DataFrame(columns=['Column','Feature','Importance Score'])
        for j in range(len(column)):
            df_features.loc[j,"Column"] = column[j]
            df_features.loc[j,"Importance Score"] = importance[j]
            if j < vocabulary.shape[0]:
                df_features.loc[j,"Feature"] = vocabulary[j]
            else:
                df_features.loc[j,"Feature"] = feature_dict[j]

        df_features.to_excel("models/features.xlsx")

    df_wrong_predictions = pd.DataFrame(columns=['Benennung (dt)','Predicted','True', 'Probability'])
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

# %%
def store_trained_model(model, sensitivity, model_folder_path):
    # save model
    model_path = model_folder_path + f"model_{str(sensitivity)[2:6]}_sensitivity.pkl"

    with open(model_path, "wb") as filestore:
        pickle.dump(model, filestore)

# %%
def evaluate_lgbm_model(model, X_test, y_test, lr_index, max_depth_index, feature_frac_index, child_index, num_models_trained, training_time, binary_model):
    probs = model.predict_proba(X_test)
    if binary_model:
        y_pred = (probs[:,1] >= lgbm_params_binary["prediction_threshold"])
        y_pred =  np.where(y_pred, 1, 0)
    else:
        y_pred = probs.argmax(axis=1)

    # Print accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, average='weighted')

    df_new = pd.DataFrame(columns=["model_name","learningrate","max_depth","num_leaves","feature_freaction","min_child_samples","accuracy", "sensitivity"])

    df_new.loc[num_models_trained, "model_name"] = f"model_{str(sensitivity)[2:6]}"
    df_new.loc[num_models_trained, "learningrate"] = lgbm_hp["lr"][lr_index]
    df_new.loc[num_models_trained, "max_depth"] = lgbm_hp["max_depth"][max_depth_index]
    df_new.loc[num_models_trained, "num_leaves"] = pow(2, lgbm_hp["max_depth"][max_depth_index])
    df_new.loc[num_models_trained, "feature_freaction"] = lgbm_hp["feature_fraction"][feature_frac_index]
    df_new.loc[num_models_trained, "min_child_samples"] = lgbm_hp["min_child_samples"][child_index]
    df_new.loc[num_models_trained, "accuracy"] = accuracy
    df_new.loc[num_models_trained, "sensitivity"] = sensitivity
    df_new.loc[num_models_trained, "Training Time (s)"] = training_time

    return y_pred, probs, accuracy, sensitivity, df_new

def store_metrics(model, X_test, y_test, evals, sensitivity, model_folder_path, binary_model):

    probs = model.predict_proba(X_test)
    if binary_model:
        y_pred = (probs[:,1] >= lgbm_params_binary["prediction_threshold"])
        y_pred =  np.where(y_pred, 1, 0)
    else:
        y_pred = probs.argmax(axis=1)

    # Print accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print("\n\n Test accuracy:", accuracy)
    print("Test Sensitivity:", sensitivity, "\n\n")

    if binary_model:
        plt.rcParams["figure.figsize"] = (10, 10)
        lgb.plot_metric(evals, metric=lgbm_params_binary["metrics"][1])
        plt.title("")
        plt.xlabel('Iterationen', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(['Training', 'Validation'], fontsize=12)
        plt.savefig(model_folder_path + 'binary_logloss_plot.png')

        plt.rcParams["figure.figsize"] = (10, 10)
        lgb.plot_metric(evals, metric=lgbm_params_binary["metrics"][0])
        plt.title("")
        plt.xlabel('Iterationen', fontsize=12 )
        plt.ylabel('AUC', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(['Training', 'Validation'], fontsize=12)
        plt.savefig(model_folder_path + 'auc_plot.png')

        class_names = ["Not relevant", "Relevant"]
        plt.rcParams["figure.figsize"] = (15, 15)
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, cmap='Blues', colorbar=False,  text_kw={'fontsize': 12})
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12 )
        plt.ylabel('True Label', fontsize=12)
        plt.savefig(model_folder_path + 'confusion_matrix.png')   

    else:    
        plt.rcParams["figure.figsize"] = (10, 10)
        lgb.plot_metric(evals, metric=lgbm_params_multiclass["metrics"][0])
        plt.title("")
        plt.xlabel('Iterationen', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(['Training', 'Validation'], fontsize=12)
        plt.savefig(model_folder_path + 'multi_logloss_plot.png')

        with open(model_folder_path + 'label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)

        class_names = []
        classes_pred = le.inverse_transform(y_pred) 
        classes_true = le.inverse_transform(y_test)

        for name in le.classes_:
            if (name in classes_pred) or (name in classes_true):
                class_names.append(name)

        cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names)
        # Passe die Größe des Diagramms an
        fig, ax = plt.subplots(figsize=(20, 25))
        # Zeige die Konfusionsmatrix an
        cm_display.plot(ax=ax, xticks_rotation='vertical', cmap='Blues', colorbar=False,  text_kw={'fontsize': 12})
        # Speichere das Diagramm
        plt.savefig(model_folder_path + 'confusion_matrix.png')

    return accuracy
