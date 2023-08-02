# %%
import pandas as pd

import os
import pickle
from loguru import logger

from src.config import paths

# %%
def load_dataset(binary_model: bool):
    data_folder = paths["folder_processed_dataset"]
    path_trainset = os.path.join(data_folder, "processed_dataset.csv")  

    if os.path.exists(path_trainset):
        df_preprocessed = pd.read_csv(path_trainset) 
    else:
        logger.error(f"No trainset found! Please check if the dataset exist at following path: {path_trainset}. If not, please use the file generate.py to create the processed dataset.")

    if binary_model:
        data_folder = data_folder + "binary/"
        train_val_test_path = os.path.join(data_folder, "binary_train_test_val_split.pkl")
        train_val_test_df_paths = os.path.join(data_folder, "binary_train_test_val_dataframes.pkl")
    else:
        data_folder = data_folder + "multiclass/"
        train_val_test_path = os.path.join(data_folder, "multiclass_train_test_val_split.pkl")
        train_val_test_df_paths = os.path.join(data_folder, "multiclass_train_test_val_dataframes.pkl")

    with open(train_val_test_path, 'rb') as handle:
        train_val_test_dict = pickle.load(handle)

    with open(train_val_test_df_paths, 'rb') as handle:
        train_val_test_df_dict = pickle.load(handle)
    
    X_train = train_val_test_dict["X_train"]
    y_train = train_val_test_dict["y_train"]
    X_val = train_val_test_dict["X_val"]
    y_val = train_val_test_dict["y_val"]
    X_test = train_val_test_dict["X_test"]
    y_test = train_val_test_dict["y_test"]
    weight_factor = train_val_test_dict["weight_factor"]

    df_test = train_val_test_df_dict["df_test"].reset_index(drop=True)

    return X_train, y_train, X_val, y_val, X_test, y_test, df_preprocessed, df_test, weight_factor

# %%
def store_trained_model(model, best_iteration, val_auc, hp, index_best_model, model_folder_path, finalmodel):
    # save model
    if finalmodel:
            model_path = model_folder_path + f"final_model.pkl"
    else:
            model_path = model_folder_path + f"model.pkl"

    with open(model_path, "wb") as filestore:
        pickle.dump(model, filestore)

    logging_file_path = model_folder_path + "logging.txt"
    if os.path.isfile(logging_file_path):
        log_text = "\nValidation AUC (final model): {}\n".format(val_auc)
        f= open(model_folder_path + "logging.txt","a")
        f.write("\n_________________________________________________\n")
        f.write(log_text)
        f.write("Trained Iterations: {}\n".format(best_iteration))
        f.close()
    else:
        dataset_path = "Dataset: {}\n".format(paths["folder_processed_dataset"])
        model_folder = "Model folder path: {}\n".format(model_folder_path)
        f= open(model_folder_path + "logging.txt","w+")
        f.write(dataset_path)
        f.write(model_folder)
        f.write("\n_________________________________________________\n")
        f.write("Best model after hyperparameter tuning:\n")
        f.write("Validation AUC: {}\n".format(val_auc))
        f.write("Trained Iterations: {}\n".format(best_iteration))
        f.write("Model index in hyperparameter tuning: {}\n".format(index_best_model+1))
        f.write("Hyperparameter:\n")
        for key in hp:
            f.write("{}: {}\n".format(key, hp[key]))
        f.close()

# %%
def store_predictions(y_test, y_pred, probs, df_preprocessed, df_test, model_folder_path, binary_model):

    if binary_model:
        class_names = df_preprocessed['Relevant fuer Messung'].unique()
    else:
        class_names = df_preprocessed["Einheitsname"].unique()
        class_names = sorted(class_names)

    df_wrong_predictions = pd.DataFrame(columns=['Sachnummer', 'Benennung (dt)', 'Derivat', 'Predicted', 'True', 'Probability'])

    try:
        y_test = y_test.to_numpy()
    except:
        pass

    # Ausgabe der Vorhersagen, der Wahrscheinlichkeiten und der wichtigsten Features
    for i in range(len(y_test)):
        try:
            if y_pred[i] != y_test[i]:
                df_wrong_predictions.loc[i,"Sachnummer"] = df_test.loc[i, "Sachnummer"]
                df_wrong_predictions.loc[i,"Benennung (dt)"] = df_test.loc[i, "Benennung (dt)"]
                df_wrong_predictions.loc[i,"Derivat"] = df_test.loc[i, "Derivat"]
                df_wrong_predictions.loc[i,"Predicted"] = class_names[y_pred[i]]
                df_wrong_predictions.loc[i,"True"] = class_names[y_test[i]]
                if binary_model:
                    if probs[i][1] >= 0.5:
                        df_wrong_predictions.loc[i,"Probability"] = probs[i][1]
                    else:
                        df_wrong_predictions.loc[i,"Probability"] = 1 - probs[i][1]
                else:
                    df_wrong_predictions.loc[i,"Probability"] = probs[i][1]
        except:
            pass
        
    # Serialize data into file:
    df_wrong_predictions.to_csv(model_folder_path + "wrong_predictions.csv")