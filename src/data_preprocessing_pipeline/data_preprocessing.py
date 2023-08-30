import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

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

from src.data_preprocessing_pipeline.feature_engineering import Feature_Engineering
from src.data_preprocessing_pipeline.data_cleaning import DataCleaner
from src.data_preprocessing_pipeline.augmentation import DataAugmention
from src.data_preprocessing_pipeline.data_analysis import store_class_distribution, analyse_data_split, store_feature_distribution
from src.utils import load_data_into_df, combine_dataframes

class DataGenerator:

    # PyTest exist
    @staticmethod
    def get_weight_factor(y, df: pd.DataFrame, binary_model: bool) -> int or dict:
        '''
        The function takes the labels (y), a pandas DataFrame, and a binary flag as input. 
        Depending on the binary flag, the function calculates and returns a weight factor either for multi-class or binary classifications. 
        For multi-class classification, it calculates the ratio of negative and positive samples for each class and returns a dictionary of weight factors. 
        For binary classification, the function calculates the ratio of samples labeled "Nein" to "Ja" and returns a single weight factor. 
        If there are no samples labeled "Ja" in the binary classification dataset, the function returns 0 and logs an error message. 
        Args: 
            y: The labels as a numpy array
            df: a pandas DataFrame object
            binary_model: binary flag (True or False) to set if it is a binary or a multiclass model
        Return: 
            weight_factor: A dictionary with keys as class labels and values as corresponding weight factors for multi-class classification or a single integer indicating the weight factor for binary classification. If there are no samples labeled "Ja" in the binary classification dataset, the function returns 0.
        '''
        if binary_model == False:
            unique_einheitsnamen = np.unique(y)
            weight_factor = {}
            for name in unique_einheitsnamen:
                weight_factor[name] = round(np.count_nonzero(y != name) / np.count_nonzero(y == name))
                if weight_factor[name] == 0:
                    weight_factor[name] = 1
        else:
            binary_label_column = config['labels']['binary_column']
            if df[df[binary_label_column]==config['labels']['binary_label_1']].shape[0] == 0:
                weight_factor = 0
                logger.error("The dataset does not contain any ""Ja"" labeled samples")
            else:
                weight_factor = round(df[df[binary_label_column]==config['labels']['binary_label_0']].shape[0] / df[df[binary_label_column]==config['labels']['binary_label_1']].shape[0])

        return weight_factor

    @staticmethod
    def train_test_val(df: pd.DataFrame, model_folder_path: str, binary_model: bool):
        '''
        This function splits the input dataframe into training, validation and test sets for binary or multiclass task. 
        The function also stores the generated sets in dictionaries. This prepares the data for the model training process.
        Args:
            df: Pandas DataFrame: The input dataframe for splitting into sets.
            model_folder_path: String: The path where different model files will be stored.
            binary_model: bool: A boolean variable indicating whether binary model will be used or multiclass model will be used for the classification task.
        Return:
            Tuple: A tuple containing X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test, weight_factor. 
                X_train, X_val and X_test respectively contain the data points of the training, validation and testing sets. 
                y_train, y_val and y_test contain the corresponding labels for the data points in X_train, X_val and X_test. 
                df_train, df_val and df_test contain the respective dataframes for the training, validation and test sets. 
                weight_factor is used as a parameter for the loss function depending on the data imbalance.
        '''
        if binary_model:
            logger.info("Split the dataset into train validation and test sets for the binary task and store the sets in dictionaries...")
        else:
            logger.info("Split the dataset into train validation and test sets for the multiclass task and store the sets in dictionaries......")
        
        # Vektorizing the text column 
        X = DataCleaner.nchar_text_to_vec(data=df, model_folder_path=model_folder_path)

        # Combine text features with bounding box features
        features = config["general_params"]["features_for_model"]
        bbox_features_dict = {"features_for_model": features}
        with open(model_folder_path + 'boundingbox_features.pkl', 'wb') as fp:
            pickle.dump(bbox_features_dict, fp)

        # Use only the text or additionally the bounding box information to generate the datasets 
        if config["general_params"]["use_only_text"] == False:
            X = np.concatenate((X, df[features].values), axis=1)

        if binary_model:
            # Map relevant features to 1 and not relevant features to 0
            y = df[config['labels']['binary_column']]
            y = y.map({config['labels']['binary_label_1']: 1, config['labels']['binary_label_0']: 0})
        else:
            # Encode the labels with the sklearn LabelEncoder
            y = df[config['labels']['multiclass_column']] 
            le = preprocessing.LabelEncoder()
            y = le.fit_transform(y)

            with open(model_folder_path + 'label_encoder.pkl', 'wb') as f: 
                pickle.dump(le, f)  

        # Get the weight factor to deal with the unbalanced dataset
        weight_factor = DataGenerator.get_weight_factor(y=y, df=df, binary_model=binary_model)     

        # Split the dataset into training, validation and testsplit
        indices = np.arange(X.shape[0])
        X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(X, y, indices, test_size=config["train_settings"]["train_val_split"], stratify=y, random_state=config["general_params"]["seed"])
        X_val, X_test, y_val, y_test, indices_val, indices_test = train_test_split(X_val, y_val, indices_val, test_size=config["train_settings"]["val_test_split"], stratify=y_val, random_state=config["general_params"]["seed"])

        # Generate dataframes of the training, validation and test sets
        df_train = df.iloc[indices_train]
        df_val = df.iloc[indices_val]
        df_test = df.iloc[indices_test]

        logger.success("Train, validation and test sets are generated!")

        return X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test, weight_factor

    @staticmethod
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

        # Load the training, validation, and test sets
        X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test, weight_factor = DataGenerator.train_test_val(df, model_folder_path=storage_path, binary_model=binary_model)

        # Analyse the dataset classes 
        analyse_data_split(df, y_train, y_val, y_test, storage_path, binary_model) 
        
        # Store the train, validation, and test datasets as arrays
        train_val_test_dict = dict({
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "weight_factor": weight_factor
        })

        # Store the train, validation, and test datasets as dataframes
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

        return train_val_test_dict, train_val_test_dataframes

    @staticmethod
    def generate_dataset() -> None:
        # Create the storage path using the current datetime
        dateTimeObj = datetime.now()
        timestamp = dateTimeObj.strftime("%d%m%Y_%H%M")
        storage_path = f"data/processed/{timestamp}/"

        # Load the labeled data into a list of dataframes
        dataframes_list, ncars = load_data_into_df(raw=False)

        # Combine all dataframes to one
        df_combined = combine_dataframes(dataframes_list, relevant_features=config["general_params"]["check_features_for_nan_values"], ncars=ncars)

        # Transdorm and add new features
        df_new_features = Feature_Engineering.add_new_features(df_combined)
        
        # Clean the dataset
        df_preprocessed, df_for_plot = DataCleaner.clean_dataset(df_new_features)

        # Generate the synthetic data (if neccessary)
        df_preprocessed = DataAugmention.data_augmentation(df_preprocessed)

        # Store the processed dataset
        os.makedirs(storage_path + "binary")
        os.makedirs(storage_path + "multiclass")
        df_preprocessed.to_csv(storage_path + "processed_dataset.csv")

        # Generate the training, validation, and test split
        train_val_test_dict_binary, train_val_test_dataframes_binary = DataGenerator.generate_dataset_dict(df_preprocessed, storage_path, binary_model=True)
        train_val_test_dict_multiclass, train_val_test_dataframes_multiclass = DataGenerator.generate_dataset_dict(df_preprocessed, storage_path, binary_model=False)

        logger.info("Generate and store the class distribution plots...")
        label_column_binary = config['labels']['binary_column']
        label_column_multiclass = config['labels']['multiclass_column']
        store_class_distribution(df_preprocessed, label_column_binary, storage_path + "binary/")
        store_class_distribution(df_preprocessed, label_column_multiclass, storage_path + "multiclass/")
        filtered_df = df_preprocessed[df_preprocessed[label_column_multiclass] != "Dummy"]
        store_class_distribution(filtered_df, label_column_multiclass, storage_path + "multiclass/")

        logger.info("Generate and store plots for feature distributions...")
        store_feature_distribution(df_preprocessed, storage_path)
        logger.success("Plots successfully stored!")

        return train_val_test_dict_binary, train_val_test_dataframes_binary, train_val_test_dict_multiclass, train_val_test_dataframes_multiclass

def main():
    DataGenerator.generate_dataset()

if __name__ == "__main__":
    
    main()