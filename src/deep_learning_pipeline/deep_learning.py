import warnings 
warnings.filterwarnings('ignore')

import pandas as pd
import math
from statistics import mean
from sklearn.metrics import recall_score, precision_score, fbeta_score

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import StratifiedKFold
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig, GatedAdditiveTreeEnsembleConfig 
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer
from pytorch_lightning.callbacks import EarlyStopping

from loguru import logger
from pathlib import Path
import sys
import time
import yaml
import json
import os
import yaml
from datetime import datetime
from yaml.loader import SafeLoader
sys.path.append(os.getcwd())

from src.utils import load_training_data

with open('src/deep_learning_pipeline/config_dl.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)


def get_metrics_results(y_true, y_pred):
    result_dict = {}
    
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    result_dict["sensitivity"] = recall_score(y_true, y_pred, average='binary')
    result_dict["precision"] = precision_score(y_true, y_pred, average='binary')
    result_dict["fbeta_score"] = fbeta_score(y_true, y_pred, beta=2, average='binary')

    return result_dict

def model_architecture(lr, dropout, batch_size, activation, layers, cat_columns, experiment_path, run_name):
    # Setting up the data configs
    data_config = DataConfig(
                                target=config["target"],
                                continuous_cols=config["continuous_cols"] + cat_columns.tolist(),
                                continuous_feature_transform=None,
                                normalize_continuous_features=config["normalize_continuous_features"]
                            )

    # Setting up trainer configs
    trainer_config = TrainerConfig(
                                    auto_lr_find=False,                                                # Runs the LRFinder to automatically derive a learning rate
                                    batch_size=batch_size,
                                    max_epochs=config["max_epochs"],
                                    early_stopping=config["earlystopping_metric"],
                                    early_stopping_min_delta=config["early_stopping_min_delta"],
                                    early_stopping_mode=config["early_stopping_mode"], 
                                    early_stopping_patience=config["early_stopping_patience"], 
                                    checkpoints=config["earlystopping_metric"], 
                                    checkpoints_path=experiment_path,
                                    load_best=True                                                  # After training, load the best checkpoint
                                )

    # Setting up optimizer configs
    optimizer_config = OptimizerConfig(optimizer='Adam')

    # Setting up model configs
    model_config = CategoryEmbeddingModelConfig(
                                                    task=config["task"],
                                                    layers=layers,  
                                                    activation=activation,
                                                    learning_rate = lr,
                                                    dropout=dropout,
                                                    loss= config["loss"],
                                                    metrics = ["fbeta_score"],
                                                    metrics_params=[{"beta":2.0}],
                                                    #metrics_prob_input=[True, True]
                                                )

    experiment_config = ExperimentConfig(project_name=experiment_path, 
                                        run_name=run_name,
                                        exp_watch="all", 
                                        log_target="tensorboard")
    # Initialize model
    tabular_model = TabularModel(
                                    data_config=data_config,
                                    model_config=model_config,
                                    optimizer_config=optimizer_config,
                                    trainer_config=trainer_config,
                                    experiment_config=experiment_config
                                )

    return tabular_model


def create_logfile(model_folder_path, data_folder, train_loss, train_fbeta_score, valid_loss, valid_fbeta_score, dropout, layers, activation, batch):
    logging_file_path = os.path.join(model_folder_path, "logging.txt")

    dataset_path = "Dataset: {}\n".format(data_folder)
    model_folder = "Model folder path: {}\n".format(model_folder_path)
    f = open(logging_file_path,"w+")
    f.write(dataset_path)
    f.write(model_folder)
    f.write("\n_________________________________________________\n")
    f.write("Results:\n")
    f.write("Training loss: {}\n".format(train_loss))
    f.write("Training Fbeta-Score: {}\n".format(train_fbeta_score))
    f.write("Validation loss: {}\n".format(valid_loss))
    f.write("Validation Fbeta-Score: {}\n".format(valid_fbeta_score))
    f.write("\n_________________________________________________\n")
    f.write("Hyperparameter:\n")
    f.write("Layer: {}\n".format(layers))
    f.write("Dropout: {}\n".format(dropout))
    f.write("Activation function: {}\n".format(activation))
    f.write("Batch Size: {}\n".format(batch))
    f.close()   

def results_from_experiment(model_path, metric):
    results_list = []
    for e in tf.compat.v1.train.summary_iterator(model_path):
        for v in e.summary.value:
            if v.tag == metric:
                results_list.append(round(v.simple_value, 5))
    
    return results_list

def get_model_results(model_path):

    result_dict = { 
                    "train_loss": [],
                    "train_fbeta_score": [],
                    "valid_loss": [],
                    "valid_fbeta_score": [],
                    "epoch": []
                }
    
    for metric in result_dict.keys():
        result_dict[metric].extend(results_from_experiment(model_path, metric))
    return result_dict

def find_event_file(folder_path):
   for file in os.listdir(folder_path):
       if file.startswith("events"):
           return os.path.join(folder_path, file)
   return None

def store_metrics(best_iteration, train_results, validation_results, metric, model_folder_path):
    plt.rcParams["figure.figsize"] = (10, 10)
    fontsize=18
    font = 'Calibri'
    plt.plot(train_results, label=f'Train {metric}')
    plt.plot(validation_results, label=f'Validation {metric}')
    plt.gca().get_lines()[0].set_color('#007F7F') 
    plt.gca().get_lines()[1].set_color('#A9D18E') 
    plt.xlabel('Epoch', fontname=font, fontsize=fontsize)
    plt.ylabel(f'{metric}', fontname=font, fontsize=fontsize)
    plt.axvline(best_iteration, color='grey', label = 'Early Stopping')
    plt.legend(['Training', 'Validation', 'Early Stopping'], fontsize=fontsize)
    plt.ylim([0, 1.2])
        
    plt.savefig(os.path.join(model_folder_path, f'{metric}_plot.png'))
    plt.close('all')

def store_confusion_matrix(df_test: pd.DataFrame, df_pred: pd.DataFrame, model_folder_path: Path):
    ''' 
    This function stores the confusion matrix of a machine learning model, given the test labels and predicted labels.
    It saves the plot in a specified folder. The function also checks whether the model is binary or multiclass, in order to design the plot and the class names. If multiclass, it loads a label encoder from a saved file. 
    Args:
        y_test: numpy array containing the true labels of the test set.
        y_pred: numpy array containing the predicted labels of the test set.
        folder_path: string containing the path where the label_encoder.pkl file is stored.
        model_folder_path: string containing the path where the resulting plot should be saved.
        binary_model: boolean variable indicating whether the model is binary or not. 
    Return: None 
    '''
    font = {'family' : 'Calibri', 'size': 30}
    colors = ['#F4FFFF', '#BBFFFF', '#00FFFF', '#00E6E6', '#00BFBF', '#009999', '#007F7F'] 
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

    class_names = ["Not relevant", "Relevant"]
    fig, ax = plt.subplots(figsize=(2, 2))
    ConfusionMatrixDisplay.from_predictions(y_true=df_test, y_pred=df_pred, display_labels=class_names, cmap=cmap, colorbar=False, text_kw=font)
    plt.xticks(fontsize=font['size'], fontname=font['family'])
    plt.yticks(fontsize=font['size'], fontname=font['family'], rotation=90, va="center")
    plt.xlabel('Predicted', font=font)
    plt.ylabel('Actual', font=font)
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder_path, 'dl_binary_confusion_matrix.png'))
    plt.close('all')

def get_training_validation_results(experiment_path, run_name):
    model_path = os.path.join(experiment_path, run_name) 
    model_path = os.path.join(model_path, "version_1")
    event_path = find_event_file(model_path)
    train_val_results_dict = get_model_results(event_path)
    
    return train_val_results_dict

def get_best_iteration(results, metric):
    metric = config["earlystopping_metric"]
    min_value = 1000
    max_value = -1
    best_iteration = 0
    for i in range(len(results[metric])):
        if config["early_stopping_mode"] == "min":
            if results[metric][i] < min_value-config["early_stopping_min_delta"]:
                min_value = results[metric][i]
                best_iteration = i   
        else:
            if results[metric][i] > max_value+config["early_stopping_min_delta"]:
                max_value = results[metric][i]
                best_iteration = i

    return best_iteration


def convert_numerical_columns(df):
    # Get the list of numerical columns
    numerical_columns = df.select_dtypes(include=[float, int]).columns

    # Convert numerical columns to float with 5 digits after the decimal point
    df[numerical_columns] = df[numerical_columns].apply(lambda x: round(x, 5))

    return df

def grid_search(df_train, df_val, df_test, cat_columns, experiment_path, timestamp):
    # Set hyperparameter
    hp_dict = config["hyperparameter"]
    model_folder = os.path.join(experiment_path, f"{timestamp}")

    # Declare the column names to generate the dataframe which stores the results and the hyperparameter of the models using grid search
    fix_columns = ["Model", "Train loss", "Train Fbeta-Score", "Validation loss", "Validation sensitivity", "Validation precision", "Validation Fbeta-Score", "Test sensitivity", "Test precision", "Test Fbeta-Score", "Training time (s)", "Trained epochs"]
    hp_columns = list(hp_dict.keys())
    columns = fix_columns + hp_columns

    # Create empty dataframe with the predefined column names
    df_gridsearch = pd.DataFrame(columns=columns)

    # Declare a dict to store the trained models
    trained_models = {}

    # Grid search hyperparameter tuning
    model_nr = 1
    hp = list(hp_dict.keys())
    logger.info("Start grid search hyperparameter tuning...")
    for hp_0 in hp_dict[hp[0]]:
        for hp_1 in hp_dict[hp[1]]:
            for hp_2 in hp_dict[hp[2]]:
                for hp_3 in hp_dict[hp[3]]:   
                    logger.info(f"Start grid search hyperparameter tuning for model {model_nr}:")
                    logger.info(f"Used hyperparameter: Layers ({hp_0}), activation functions ({hp_1}), batch size ({hp_2}), dropout ({hp_3}) ")
                    # Define the name for the run
                    run_name = f"{timestamp}/grid_search/model_{model_nr}/layer_{hp_0}_activaten_{hp_1}_batch_{hp_2}_dr_{hp_3}"

                    # Build the model archtecture
                    tabular_model = model_architecture(lr= 0.0001, dropout=hp_3, batch_size=hp_2, activation=hp_1, layers=hp_0, cat_columns=cat_columns, experiment_path=experiment_path, run_name=run_name)

                    # Train and evaluate the model
                    start = time.time()
                    tabular_model.fit(train=df_train, validation=df_val)
                    stop = time.time()
                    training_time = int(stop - start)

                    # Get training and validation results
                    train_val_results_dict = get_training_validation_results(experiment_path, run_name)

                    # Get predictions and results on the validation set
                    pred_df = tabular_model.predict(df_val)
                    validation_results = get_metrics_results(y_true=pred_df["Relevant fuer Messung"], y_pred=pred_df["prediction"])
                                        
                    # Get predictions and results on the testset
                    pred_df_test = tabular_model.predict(df_test)
                    test_results = get_metrics_results(y_true=pred_df_test["Relevant fuer Messung"], y_pred=pred_df_test["prediction"])
                    
                    # Get the index of the best iteration 
                    best_iteration = get_best_iteration(results=train_val_results_dict, metric=config["earlystopping_metric"])

                    # Add model and results to list of trained models
                    str_model_nr = "model_nr " + str(model_nr)
                    trained_models[str_model_nr] = []
                    trained_models[str_model_nr].append(tabular_model)
                    trained_models[str_model_nr].append(train_val_results_dict)
                    trained_models[str_model_nr].append(best_iteration)

                    # Store the model results in a dataframe
                    epochs_trained = len(train_val_results_dict["valid_loss"])-1
                    df_gridsearch.loc[model_nr, "Model"] = model_nr

                    # Try except, because an error in saving the train loss. The number of train loss results are less than epochs are trained
                    try:
                        df_gridsearch.loc[model_nr, "Train loss"] = train_val_results_dict["train_loss"][best_iteration]
                    except:
                        df_gridsearch.loc[model_nr, "Train loss"] = train_val_results_dict["train_loss"][-1]
                    df_gridsearch.loc[model_nr, "Train Fbeta-Score"] = train_val_results_dict["train_fbeta_score"][best_iteration]
                    
                    df_gridsearch.loc[model_nr, "Validation loss"] = train_val_results_dict["valid_loss"][best_iteration]
                    df_gridsearch.loc[model_nr, "Validation sensitivity"] = validation_results["sensitivity"]
                    df_gridsearch.loc[model_nr, "Validation precision"] = validation_results["precision"]
                    df_gridsearch.loc[model_nr, "Validation Fbeta-Score"] = validation_results["fbeta_score"]            

                    df_gridsearch.loc[model_nr, "Test sensitivity"] = test_results["sensitivity"]
                    df_gridsearch.loc[model_nr, "Test precision"] = test_results["precision"]     
                    df_gridsearch.loc[model_nr, "Test Fbeta-Score"] = test_results["fbeta_score"]            

                    df_gridsearch.loc[model_nr, "Training time (s)"] = training_time
                    df_gridsearch.loc[model_nr, "Trained epochs"] = int(train_val_results_dict["epoch"][-1])

                    df_gridsearch.loc[model_nr, hp_columns[0]] = str(hp_0)
                    df_gridsearch.loc[model_nr, hp_columns[1]] = str(hp_1)
                    df_gridsearch.loc[model_nr, hp_columns[2]] = hp_2
                    df_gridsearch.loc[model_nr, hp_columns[3]] = hp_3

                    valid_loss = train_val_results_dict["valid_loss"][best_iteration]
                    valid_fbeta_score = train_val_results_dict["valid_fbeta_score"][best_iteration]

                    # Store the results of the trained models after grid search
                    df_gridsearch.to_csv(os.path.join(model_folder, "grid_search/results_grid_search.csv"))
                    logger.info(f"Model {model_nr} successfully trained. Early stopping in epoch {best_iteration}/{epochs_trained} - with {valid_loss} valid loss and {valid_fbeta_score} valid f1-score \n")
                    model_nr = model_nr + 1


    logger.success("Hyperparametertuning was successfull!")

    # Sort the results by the validation Fbeta-score in descening order 
    df_gridsearch_sorted = df_gridsearch.sort_values(by=["Validation Fbeta-Score", "Validation loss"], ascending=[False, True])

    df_gridsearch_sorted_converted = convert_numerical_columns(df_gridsearch_sorted)

    # Store the results of the trained models after grid search
    df_gridsearch_sorted_converted.to_csv(os.path.join(model_folder, "grid_search/results_grid_search.csv"))

    return trained_models, model_folder, df_gridsearch_sorted_converted

def k_fold_crossvalidation(df_train_cv, df_gridsearch_sorted, cat_columns, experiment_path, timestamp):
    
    # Declare the column names to generate the dataframe which stores the results and the hyperparameter of the models using grid search
    fix_columns = ["Model", "Avg train loss", "Avg train Fbeta-Score", "Avg validation loss", "Avg validation sensitivity", "Avg validation precision", "Avg validation Fbeta-Score"]
 
    hp_columns = list(config["hyperparameter"].keys())
    columns = fix_columns + hp_columns

    model_folder = os.path.join(experiment_path, f"{timestamp}")

    # Create empty dataframe with the predefined column names
    results_cv = pd.DataFrame(columns=columns)

    # Get the number of models which should be used for cross-validation
    top_x_models = int(round(df_gridsearch_sorted.shape[0] * config["top_x_models_for_cv"]))
    if top_x_models == 0:
        top_x_models = 1

    # Set the number of folds 
    number_of_folds = config["k-folds"]

    # Total number of folds
    total_cv_models = number_of_folds * top_x_models

    # Set Stratified K-Fold cross-validator
    kfold = StratifiedKFold(n_splits=config["k-folds"], shuffle=True, random_state=42)

    # Cross-Validation
    logger.info(f"Start to validate the top {top_x_models} models by using {number_of_folds}-fold cross-validation... ")
    for i in range(top_x_models):
        # Get the hyperparameter according to the sorted list (F1-Score descending order) of results after grid search hyperparameter tuning 
        dropout = float(df_gridsearch_sorted.iloc[i]["dropout"])
        batch = int(df_gridsearch_sorted.iloc[i]["batch_size"])
        activation = df_gridsearch_sorted.iloc[i]["activation_functions"]
        layers = df_gridsearch_sorted.iloc[i]["layers"]

        if dropout == 0.0:
            dropout = 0
        # Declare empty lists to store the training and validation results
        train_loss = []
        train_fbeta_score = []
        valid_loss = []
        valid_fbeta_score = []
        valid_sensitivity = []
        valid_precision = []

        # K-Fold Cross-Validation for each fold
        logger.info(f"Start {number_of_folds}-fold cross-validation for model {i+1}/{top_x_models}:")
        for fold, (train_idx, val_idx) in enumerate(kfold.split(df_train_cv, df_train_cv[config["target"]])):
            logger.info(f"Fold: {fold+1}/{number_of_folds}")
            
            # Define the name for the run
            run_name = f"{timestamp}/cross_validation/model_{i+1}/fold{fold+1}_layer_{layers}_activaten_{activation}_batch_{batch}_dr_{dropout}"

            # Get the train and validaiton data for the current fold
            train_fold = df_train_cv.iloc[train_idx]
            val_fold = df_train_cv.iloc[val_idx]

            # Initialize the tabular model
            tabular_model = model_architecture(lr= 0.0001, dropout=dropout, batch_size=batch, activation=activation, layers=layers, cat_columns=cat_columns, experiment_path=experiment_path, run_name=run_name)

            # Fit the model
            tabular_model.fit(train=train_fold, validation=val_fold)
            
            # Get results after training
            train_val_results_dict = get_training_validation_results(experiment_path, run_name)

            # Get index of best iteration 
            best_iteration = get_best_iteration(results=train_val_results_dict, metric=config["earlystopping_metric"])

            # Get predictions and results on the validation set
            pred_df = tabular_model.predict(val_fold)
            validation_results = get_metrics_results(y_true=pred_df["Relevant fuer Messung"], y_pred=pred_df["prediction"])

            # Store confusion matrix
            binary_model_path = os.path.join(experiment_path, run_name)
            store_confusion_matrix(df_test=pred_df["Relevant fuer Messung"], df_pred=pred_df["prediction"], model_folder_path=binary_model_path)

            # Append metric results of the current fold
            try:
                train_loss.append(train_val_results_dict["train_loss"][best_iteration])
            except:
                train_loss.append(train_val_results_dict["train_loss"][-1])

            train_fbeta_score.append(train_val_results_dict["train_fbeta_score"][best_iteration])
            valid_loss.append(train_val_results_dict["valid_loss"][best_iteration])
            valid_sensitivity.append(validation_results["sensitivity"])
            valid_fbeta_score.append(validation_results["fbeta_score"])
            valid_precision.append(validation_results["precision"])

            #print("Trainvalresultsdict:\n")
            #print(train_val_results_dict)
            #print("validation_results:\n")
            #print(validation_results)

        #print("Valid Loss:\n")
        #print(valid_loss)
        #print("Valid Fbeta:\n")
        #print(valid_fbeta_score)

        # Store results in a dataframe
        results_cv.loc[i, "Model"] = df_gridsearch_sorted.iloc[i]["Model"]
        results_cv.loc[i, "Avg train loss"] = round(mean(train_loss), 6)
        results_cv.loc[i, "Avg train Fbeta-Score"] = round(mean(train_fbeta_score), 6)
        results_cv.loc[i, "Avg validation loss"] = round(mean(valid_loss), 6)
        results_cv.loc[i, "Avg validation sensitivity"] = round(mean(valid_sensitivity), 6)
        results_cv.loc[i, "Avg validation Fbeta-Score"] = round(mean(valid_fbeta_score), 6)
        results_cv.loc[i, "Avg validation precision"] = round(mean(valid_precision), 6)
        
        results_cv.loc[i, "layers"] = str(layers)
        results_cv.loc[i, "dropout"] = dropout
        results_cv.loc[i, "batch_size"] = batch
        results_cv.loc[i, "activation_functions"] = str(activation)
    
    # Sort results by the average validation f1-score
    results_cv_sorted = results_cv.sort_values(by=["Avg validation Fbeta-Score", "Avg validation loss"], ascending=[False, True])
    best_model_index = results_cv_sorted.iloc[0]["Model"]

    # Assuming you have a DataFrame called 'df'
    df_cv_sorted_converted = convert_numerical_columns(results_cv_sorted)

    # Store results in a csv file
    df_cv_sorted_converted.to_csv(os.path.join(model_folder, "cross_validation/results_cross_validation.csv"))
        
    logger.success("Cross-Validation was successfull!")

    return best_model_index, results_cv_sorted

def feature_selection(df_train, df_val, df_test):

    # Select relevant features
    relevant_columns = config["continuous_cols"] + config["text_cols"] + config["target"]
    df_train = df_train[relevant_columns]
    df_val = df_val[relevant_columns]
    df_test = df_test[relevant_columns]

    # Map relevant car parts to 1 and not relevant car parts to 0
    label_1 = config['target_classes'][0]
    label_0 = config['target_classes'][1]
    target_column = config['target'][0]
    df_train[target_column] = df_train[target_column].map({label_1:1, label_0:0}).astype("category")
    df_val[target_column] = df_val[target_column].map({label_1:1, label_0:0}).astype("category")
    df_test[target_column] = df_test[target_column].map({label_1:1, label_0:0}).astype("category")

    '''
    # Convert car part designation to category
    df_train[cat_column] = df_train[cat_column].astype("category").cat.codes
    df_val[cat_column] = df_val[cat_column].astype("category").cat.codes
    df_test[cat_column] = df_test[cat_column].astype("category").cat.codes
    '''

    return df_train, df_val, df_test

def train():
    # Define the experiment directory and get the starting date and time 
    dateTimeObj = datetime.now()
    timestamp = dateTimeObj.strftime("%Y%m%d_%H%M")
    experiment_path = Path(config["experiment_path"])

    # Load the training, validation and test data
    X_train, y_train, X_val, y_val, X_test, y_test, df_preprocessed, df_train, df_val, df_test, weight_factor = load_training_data(data_folder=config["folder_processed_dataset"], binary_model=True)
    
    # Feature selection
    df_train, df_val, df_test = feature_selection(df_train, df_val, df_test)

    # Fit vectorizer
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 8), lowercase=False)
    relevant_car_parts = df_preprocessed[df_preprocessed["Relevant fuer Messung"] == "Ja"]
    vectorizer.fit(relevant_car_parts['Benennung (bereinigt)'])
    cat_columns=vectorizer.get_feature_names_out()

    # Transform the text in column "Benennung (bereinigt)"  in the "df_train" DataFrame using the trained vectorizer
    transformed_data = vectorizer.transform(df_train['Benennung (bereinigt)'])
    transformed_df = pd.DataFrame(transformed_data.toarray(), columns=vectorizer.get_feature_names_out()).astype("category")
    df_train = pd.concat([df_train.drop('Benennung (bereinigt)', axis=1), transformed_df], axis=1)

    # Transform the text in column "Benennung (bereinigt)"  in the "df_train" DataFrame using the trained vectorizer
    transformed_data = vectorizer.transform(df_val['Benennung (bereinigt)'])
    transformed_df = pd.DataFrame(transformed_data.toarray(), columns=vectorizer.get_feature_names_out()).astype("category")
    df_val = pd.concat([df_val.drop('Benennung (bereinigt)', axis=1), transformed_df], axis=1)

    # Transform the text in column "Benennung (bereinigt)"  in the "df_train" DataFrame using the trained vectorizer
    transformed_data = vectorizer.transform(df_test['Benennung (bereinigt)'])
    transformed_df = pd.DataFrame(transformed_data.toarray(), columns=vectorizer.get_feature_names_out()).astype("category")
    df_test = pd.concat([df_test.drop('Benennung (bereinigt)', axis=1), transformed_df], axis=1)

    # Start grid search hyperparametertuning
    trained_models, model_folder, df_gridsearch_sorted = grid_search(df_train, df_val, df_test, cat_columns, experiment_path, timestamp)
    
    # Combine training and validation set for cross-validation
    df_train_cv = pd.concat([df_train, df_val], ignore_index=True)

    # Start k-fold cross-validation
    best_model_index, results_cv_sorted = k_fold_crossvalidation(df_train_cv, df_gridsearch_sorted, cat_columns, experiment_path, timestamp)

    # Get the model with the highest average validation f1-score after cross-validation
    model_name = "model_nr " + str(best_model_index)
    best_model_after_crossvalidation = trained_models[model_name][0]

    # Store the best model after hyperparameter tuning and crossvalidation
    binary_model_path = os.path.join(model_folder, "best_binary_model_after_cv")
    best_model_after_crossvalidation.save_model(binary_model_path)

    best_iteration = trained_models[model_name][2]

    # Store the loss plot
    train_loss = trained_models[model_name][1]["train_loss"]
    valid_loss = trained_models[model_name][1]["valid_loss"]
    store_metrics(best_iteration=best_iteration, train_results=train_loss, validation_results=valid_loss, metric="Loss", model_folder_path=binary_model_path)

    # Store the Fbeta-Score plot
    train_fbeta_score = trained_models[model_name][1]["train_fbeta_score"]
    valid_fbeta_score = trained_models[model_name][1]["valid_fbeta_score"]
    store_metrics(best_iteration=best_iteration, train_results=train_fbeta_score, validation_results=valid_fbeta_score, metric="Fbeta-Score", model_folder_path=binary_model_path)


    # Make predictions on the test set
    pred_df = best_model_after_crossvalidation.predict(df_test)

    # Store wrong predictions on the test set
    df_filtered = pred_df[pred_df["Relevant fuer Messung"] != pred_df["prediction"]]
    df_filtered.to_excel(os.path.join(binary_model_path, "wrong_predictions.xlsx"))

    # Store confusion matrix
    store_confusion_matrix(df_test=pred_df["Relevant fuer Messung"], df_pred=pred_df["prediction"], model_folder_path=binary_model_path)

    # Train and store the final model 
    # Combine train and test set to a new train set for training the final model on a larger dataset
    df_train_final_model = pd.concat([df_train, df_test], ignore_index=True)

    # Get the hyperparameter of the model with the highest validation f1-score after crossvalidation
    dropout = results_cv_sorted.iloc[0]["dropout"]
    batch = results_cv_sorted.iloc[0]["batch_size"]
    activation = results_cv_sorted.iloc[0]["activation_functions"]
    layers = results_cv_sorted.iloc[0]["layers"]

    # Define the run name
    run_name = f"{timestamp}/final_binary_model/layer_{layers}_activaten_{activation}_batch_{batch}_dr_{dropout}"

    # Define the model architecture
    final_model = model_architecture(lr= 0.0001, dropout=dropout, batch_size=batch, activation=activation, layers=layers, cat_columns=cat_columns, experiment_path=experiment_path, run_name=run_name)
    
    # Train and evaluate the final model
    final_model.fit(train=df_train_final_model, validation=df_val)

    # Get results after training
    train_val_results_dict = get_training_validation_results(experiment_path, run_name)

    # Get index of best iteration 
    best_iteration = get_best_iteration(results=train_val_results_dict, metric=config["earlystopping_metric"])
    
    # Get the metric results of the best iteration to store them in a logfile
    try:
        train_loss = train_val_results_dict["train_loss"][best_iteration]
    except:
        train_loss = 0
    train_fbeta_score = train_val_results_dict["train_fbeta_score"][best_iteration]
    valid_loss = train_val_results_dict["valid_loss"][best_iteration]
    valid_fbeta_score = train_val_results_dict["valid_fbeta_score"][best_iteration]

    # Define the target directory of the final model
    final_model_path = os.path.join(experiment_path, f"{timestamp}/final_binary_model/model_1")

    # Store the final model
    final_model.save_model(final_model_path)

    # Create the logfile
    create_logfile(final_model_path, config["folder_processed_dataset"], train_loss, train_fbeta_score, valid_loss, valid_fbeta_score, dropout, layers, activation, batch)

    # Store the loss plot
    store_metrics(best_iteration=best_iteration, train_results=train_val_results_dict["train_loss"], validation_results=train_val_results_dict["valid_loss"], metric="Loss", model_folder_path=final_model_path)

    # Store the Fbeta-Score plot
    store_metrics(best_iteration=best_iteration, train_results=train_val_results_dict["train_fbeta_score"], validation_results=train_val_results_dict["valid_fbeta_score"], metric="Fbeta-Score", model_folder_path=final_model_path)

    # Store the model summary to pkl file
    #tabular_model.summary()

# %%
if __name__ == "__main__":
    
    train()

