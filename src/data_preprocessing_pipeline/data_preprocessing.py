import pandas as pd

import os
import pickle
from datetime import datetime
from loguru import logger

import yaml
from yaml.loader import SafeLoader
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

import sys
sys.path.append(config['paths']['project_path'])

from src.utils import load_data_into_df, combine_dataframes
from src.data_preprocessing_pipeline.feature_engineering import add_new_features
from src.data_preprocessing_pipeline.data_cleaning import clean_dataset, train_test_val
from src.data_preprocessing_pipeline.augmentation import data_augmentation
from src.data_preprocessing_pipeline.data_analysis import store_class_distribution, analyse_data_split, plot_distribution

def generate_dataset_dict(df: pd.DataFrame, storage_path: str, binary_model: bool) -> None:
    '''
    This function takes a pandas DataFrame containing the dataset, path of the folder where the train, validation and test splits will be stored, and a boolean value indicating whether the model is binary or not. 
    The function does the following:
        Calls train_test_val() function on the input DataFrame to split the dataset into train, validation and test sets.
        Calls analyse_data_split() function to analyze and visualize the distribution of data in the train, validation and test splits.
        Stores the train, validation and test splits as dictionaries with the keys "X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "weight_factor" in the path specified by the input argument "storage_path".
        Stores the pandas dataframes of the train, validation and test splits as dictionaries in the path specified by the input argument "storage_path". 
    Args:
        df: a pandas DataFrame which contains the dataset to be split
        storage_path: a string which contains the path to the folder where the train, validation and test splits will be stored
        binary_model: a boolean which indicates if the model is binary or not 
    Return: None 
    '''

    X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test, weight_factor = train_test_val(df, model_folder_path=storage_path, binary_model=binary_model)

    analyse_data_split(df, y_train, y_val, y_test, storage_path, binary_model) 
    
    train_val_test_dict = dict({
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "weight_factor": weight_factor
    })

    train_val_test_dataframes = dict({
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test
    })

    if binary_model:
        with open(storage_path + 'binary/binary_train_test_val_split.pkl', 'wb') as handle:
            pickle.dump(train_val_test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(storage_path + 'binary/binary_train_test_val_dataframes.pkl', 'wb') as handle:
            pickle.dump(train_val_test_dataframes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(storage_path + 'multiclass/multiclass_train_test_val_split.pkl', 'wb') as handle:
            pickle.dump(train_val_test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(storage_path + 'multiclass/multiclass_train_test_val_dataframes.pkl', 'wb') as handle:
            pickle.dump(train_val_test_dataframes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.success("Splitted datasets are successfully stored!")

def drop_unnamed_columns(df):
   unnamed_cols = [col for col in df.columns if col.startswith('Unnamed')]
   df.drop(unnamed_cols, axis=1, inplace=True)
   return df

def generate_dataset() -> None:
    dateTimeObj = datetime.now()
    timestamp = dateTimeObj.strftime("%d%m%Y_%H%M")
    storage_path = f"data/processed/{timestamp}/"

    dataframes_list, ncars = load_data_into_df()

    df_combined = combine_dataframes(dataframes_list, relevant_features=config["general_params"]["check_features_for_nan_values"], ncars=ncars)

    df_new_features = add_new_features(df_combined)
    
    df_preprocessed, df_for_plot = clean_dataset(df_new_features)

    df_preprocessed = drop_unnamed_columns(df_preprocessed)

    # Generate the new dataset
    df_preprocessed = data_augmentation(df_preprocessed)

    os.makedirs(storage_path + "binary")
    os.makedirs(storage_path + "multiclass")
    df_preprocessed.to_csv(storage_path + "processed_dataset.csv")

    generate_dataset_dict(df_preprocessed, storage_path, binary_model=True)
    generate_dataset_dict(df_preprocessed, storage_path, binary_model=False)

    logger.info("Generate and store the class distribution plots...")
    label_column_binary = config['labels']['binary_column']
    label_column_multiclass = config['labels']['multiclass_column']
    store_class_distribution(df_preprocessed, label_column_binary, storage_path + "binary/")
    store_class_distribution(df_preprocessed, label_column_multiclass, storage_path + "multiclass/")
    filtered_df = df_preprocessed[df_preprocessed[label_column_multiclass] != "Dummy"]
    store_class_distribution(filtered_df, label_column_multiclass, storage_path + "multiclass/")

    logger.info("Generate and store plots for feature distributions...")
    plot_distribution(df_preprocessed, storage_path)
    logger.success("Plots successfully stored!")

def main():
    generate_dataset()

if __name__ == "__main__":
    
    main()