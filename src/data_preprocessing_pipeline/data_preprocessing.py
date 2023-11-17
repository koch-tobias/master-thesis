import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from datetime import datetime

from loguru import logger
from pathlib import Path
import pickle
import os
import sys
import yaml
from yaml.loader import SafeLoader
sys.path.append(os.getcwd())

from src.data_preparation_pipeline.data_preparation import Preperator
from src.data_preprocessing_pipeline.feature_engineering import Feature_Engineering
from src.data_preprocessing_pipeline.data_cleaning import DataCleaner
from src.data_preprocessing_pipeline.augmentation import DataAugmention
from src.data_preprocessing_pipeline.data_analysis import store_class_distribution, analyse_data_split, store_feature_distribution
from src.utils import load_data_into_df, combine_dataframes

with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

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
            weight_factor = {}
            if y.size != 0:
                unique_einheitsnamen, counts = np.unique(y, return_counts=True)
                max_records_most_common_class = max(counts)
        
                for name in unique_einheitsnamen:
                    weight_factor[name] = round(max_records_most_common_class / np.count_nonzero(y == name))
                    if weight_factor[name] == 0:
                        weight_factor[name] = 1
        else:
            binary_label_column = config['labels']['binary_column']
            if df[df[binary_label_column]==config['labels']['binary_label_0']].shape[0] == 0:
                weight_factor = 0
                logger.error("The dataset does not contain any ""Ja"" labeled samples")
            else:
                weight_factor = round(df[df[binary_label_column]==config['labels']['binary_label_1']].shape[0] / df[df[binary_label_column]==config['labels']['binary_label_0']].shape[0])

        return weight_factor

    @staticmethod
    def train_test_val(df: pd.DataFrame, model_folder_path: Path, binary_model: bool):
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
        features = config["dataset_params"]["features_for_model"]
        bbox_features_dict = {"features_for_model": features}
        with open(os.path.join(model_folder_path, 'boundingbox_features.pkl'), 'wb') as fp:
            pickle.dump(bbox_features_dict, fp)

        # Use only the text or additionally the bounding box information to generate the datasets 
        if config["dataset_params"]["use_only_text"] == False:
            X = np.concatenate((X, df[features].values), axis=1)

        if binary_model:
            # Map relevant features to 0 and not relevant features to 1
            y = df[config['labels']['binary_column']]
            y = y.map({config['labels']['binary_label_1']: 1, config['labels']['binary_label_0']: 0})
        else:
            # Encode the labels with the sklearn LabelEncoder
            y = df[config['labels']['multiclass_column']] 
            le = preprocessing.LabelEncoder()
            y = le.fit_transform(y)

            with open(os.path.join(model_folder_path, 'label_encoder.pkl'), 'wb') as f: 
                pickle.dump(le, f)  

        # Get the weight factor to deal with the unbalanced dataset
        weight_factor = DataGenerator.get_weight_factor(y=y, df=df, binary_model=binary_model)     

        # Split the dataset into training, validation and testsplit
        indices = np.arange(X.shape[0])
        X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(X, y, indices, test_size=config["dataset_params"]["train_val_split"], stratify=y, random_state=config["dataset_params"]["seed"])
        X_val, X_test, y_val, y_test, indices_val, indices_test = train_test_split(X_val, y_val, indices_val, test_size=config["dataset_params"]["val_test_split"], stratify=y_val, random_state=config["dataset_params"]["seed"])

        # Generate dataframes of the training, validation and test sets
        df_train = df.iloc[indices_train]
        df_val = df.iloc[indices_val]
        df_test = df.iloc[indices_test]

        logger.success("Train, validation and test sets are generated!")

        return X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test, weight_factor

    @staticmethod
    def generate_dataset_dict(df: pd.DataFrame, storage_path: Path, binary_model: bool) -> None:
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

        """
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
        """
        
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
            with open(os.path.join(storage_path, 'binary/binary_train_test_val_split.pkl'), 'wb') as handle:
                pickle.dump(train_val_test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            with open(os.path.join(storage_path, 'binary/binary_train_test_val_dataframes.pkl'), 'wb') as handle:
                pickle.dump(train_val_test_dataframes, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join(storage_path, 'multiclass/multiclass_train_test_val_split.pkl'), 'wb') as handle:
                pickle.dump(train_val_test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(storage_path, 'multiclass/multiclass_train_test_val_dataframes.pkl'), 'wb') as handle:
                pickle.dump(train_val_test_dataframes, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logger.success("Splitted datasets are successfully stored!")

        return train_val_test_dict, train_val_test_dataframes

    @staticmethod
    def normalize_dataframe(df):
        # select only numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        # create StandardScaler object to normalize the numeric columns
        scaler = StandardScaler()

        # normalize the selected columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        return df
    
    @staticmethod
    def generate_dataset() -> None:
        # Create the storage path using the current datetime
        dateTimeObj = datetime.now()
        timestamp = dateTimeObj.strftime("%Y%m%d_%H%M")
        storage_path = Path(f"data/processed/{timestamp}/")

        binary_path = os.path.join(storage_path, "binary")
        multiclass_path = os.path.join(storage_path, "multiclass")

        os.makedirs(binary_path)
        os.makedirs(multiclass_path)

        # Load the labeled data into a list of dataframes
        dataframes_list, ncars = load_data_into_df(raw=False)

        logger.info("Start preparing the data...")
        for index, dataframe in enumerate(dataframes_list) :
            dataframes_list[index], ncar = Preperator.data_preparation(dataframe)
        logger.success(f"The data is successfully prepared! The features are reduced and formated to the correct data type, subfolders are deleted, and only relevant modules are kept!")

        # Combine all dataframes to one
        df_combined = combine_dataframes(dataframes_list, relevant_features=config["dataset_params"]["bounding_box_features_original"], ncars=ncars)
    
        # Transdorm and add new features
        df_new_features = Feature_Engineering.add_new_features(df_combined)

        store_class_distribution(df_new_features, config['labels']['binary_column'], binary_path)
        store_class_distribution(df_new_features, config['labels']['multiclass_column'], multiclass_path)

        # Clean the dataset
        df_preprocessed, df_for_plot = DataCleaner.clean_dataset(df_new_features)

        # Generate the synthetic data (if neccessary)
        df_preprocessed = DataAugmention.data_augmentation(df_preprocessed)

        # Normalize the numerical features
        if config["dataset_params"]["normalize_numerical_features"]:
            df_preprocessed = DataGenerator.normalize_dataframe(df_preprocessed)

        # Store the processed dataset
        df_preprocessed.to_csv(os.path.join(storage_path, "processed_dataset.csv"))

        # Get label column names
        label_column_binary = config['labels']['binary_column']
        label_column_multiclass = config['labels']['multiclass_column']

        # Get data with only relevant components
        df_only_relevants = df_preprocessed[df_preprocessed[label_column_binary] == "Ja"]
        df_only_relevants.to_csv(os.path.join(storage_path, "processed_dataset_only_relevants.csv"))

        # Generate the training, validation, and test split
        train_val_test_dict_binary, train_val_test_dataframes_binary = DataGenerator.generate_dataset_dict(df_preprocessed, storage_path, binary_model=True)
        train_val_test_dict_multiclass, train_val_test_dataframes_multiclass = DataGenerator.generate_dataset_dict(df_only_relevants, storage_path, binary_model=False)

        logger.info("Generate and store the class distribution plots...")

        store_class_distribution(df_preprocessed, label_column_binary, binary_path)
        store_class_distribution(df_only_relevants, label_column_multiclass, multiclass_path)
        #filtered_df = df_preprocessed[df_preprocessed[label_column_multiclass] != "Dummy"]
        #store_class_distribution(filtered_df, label_column_multiclass, multiclass_path)

        logger.info("Generate and store plots for feature distributions...")
        store_feature_distribution(df_preprocessed, storage_path)
        logger.success("Plots successfully stored!")

        return train_val_test_dict_binary, train_val_test_dataframes_binary, train_val_test_dict_multiclass, train_val_test_dataframes_multiclass

def main():
    DataGenerator.generate_dataset()

if __name__ == "__main__":
    
    main()