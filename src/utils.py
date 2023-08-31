# %%
import pandas as pd

import os
import pickle
import json
from loguru import logger

import yaml
from yaml.loader import SafeLoader
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

def load_training_data(binary_model: bool):
    ''' 
    This function loads the preprocessed dataset along with train, validation and test sets from the specified data folder. The binary_model parameter indicates whether the dataset is for a binary classification task or not.
    Args:
        binary_model: A boolean value indicating whether the dataset is for a binary classification task or not.
    Return:
        X_train: An array or DataFrame containing the training set features.
        y_train: An array or DataFrame containing the training set labels.
        X_val: An array or DataFrame containing the validation set features.
        y_val: An array or DataFrame containing the validation set labels.
        X_test: An array or DataFrame containing the test set features.
        y_test: An array or DataFrame containing the test set labels.
        df_preprocessed: The preprocessed dataset.
        df_test: A DataFrame containing the test set data.
        weight_factor: A float value indicating the weight factor for the dataset.
    '''
    # Get dataset path from config.yaml
    data_folder = config["train_settings"]["folder_processed_dataset"]
    path_trainset = os.path.join(data_folder, "processed_dataset.csv")  

    # Load dataset
    if os.path.exists(path_trainset):
        df_preprocessed = pd.read_csv(path_trainset) 
    else:
        logger.error(f"No trainset found! Please check if the dataset exist at following path: {path_trainset}. If not, please use the file generate.py to create the processed dataset.")

    # Create paths to load the datasets
    if binary_model:
        train_val_test_path = os.path.join(data_folder, "binary/binary_train_test_val_split.pkl")
        train_val_test_df_paths = os.path.join(data_folder, "binary/binary_train_test_val_dataframes.pkl")
    else:
        train_val_test_path = os.path.join(data_folder, "multiclass/multiclass_train_test_val_split.pkl")
        train_val_test_df_paths = os.path.join(data_folder, "multiclass/multiclass_train_test_val_dataframes.pkl")

    # Load and return the datasets
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

    df_train = train_val_test_df_dict["df_test"].reset_index(drop=True)
    df_val = train_val_test_df_dict["df_test"].reset_index(drop=True)
    df_test = train_val_test_df_dict["df_test"].reset_index(drop=True)

    return X_train, y_train, X_val, y_val, X_test, y_test, df_preprocessed, df_train, df_val, df_test, weight_factor

def prepare_columns(df):
    ''' 
    This function prepares the columns of a DataFrame by removing double quotes and replacing commas with dots in values that start with a double quote.
    This is important that the labeled data can stored as .csv later
    Args:
        df: A pandas DataFrame object.
    Return:
        df: The modified DataFrame with the desired modifications.
    '''
    # Iterate over the columns in the DataFrame
    for column in df.columns:
        # Iterate over the values in the specified column
        for i, value in enumerate(df[column]):
            # Check if the value starts with a double quote
            if str(value).startswith('"'):
                # Remove the double quotes and replace commas with dots
                modified_value = str(value).replace('"', '').replace(',', '.')
                # Update the value in the DataFrame
                df.at[i+1, column] = modified_value
    
    return df

def read_file(file_path, raw: bool):
    if raw:
        # Load the excel into a pandas dataframe, delete the header and declare the second row as new header

        df = pd.read_excel(file_path, header=None, skiprows=1)
        df.columns = df.iloc[0]
        df = df.iloc[1:]

        # Drop all empty columns
        df = df.dropna(how= "all", axis=1, inplace=False)

        # Store the ncar abbreviation for file paths
        ncar = df.iloc[0]['Code']

        df = prepare_columns(df)
    
    else:
        df = pd.read_csv(file_path)
        ncar = file_path.split("_")[0]
        df["Derivat"] = ncar
    
    return df, ncar

def load_data_into_df(raw: bool) -> tuple[list, str]:
    ''' 
    This function loads data from the specified folder path. It reads data from all files in the folder, converts them to pandas dataframes and stores the dataframes in a list. 
    The list of dataframes and a list of associated NCAR codes are returned as outputs. 
    Args:
        None
    Return:
        dataframes: a list containing pandas dataframes of all the files read from the specified folder path
        ncars: a list of the associated NCAR codes for all the files in the dataframes 
    '''

    if raw:
        folder_name = "data/raw_for_labeling"
    else:
        folder_name = "data/labeled"

    # Check if the folder exists   
    if not os.path.exists(folder_name):
        logger.error(f"The path {folder_name} does not exist.")
        exit()
    else:
        logger.info("Loading the labeled datasets...")

        # Create an empty list to store all dataframes
        dataframes = []
        ncars = []
        # Loop through all files in the folder and open them as dataframes
        for file in os.listdir(folder_name):
                try:
                    file_path = os.path.join(folder_name, file)
                    df, ncar = read_file(file_path, raw)
                except:
                    logger.info(f"Error reading file {file}. Skipping...")
                    continue

                # Add the created dataframe to the list of dataframes
                dataframes.append(df)
                ncars.append(ncar)


    # Check if any dataframes were created
    if len(dataframes) == 0:
        logger.error(f"No dataframes were created - please check if the files in folder {folder_name} are correct/exist.")
        exit()
    else:
        logger.success(f"{len(dataframes)} dataframe(s) were created.")

        return dataframes, ncars
    
def check_nan_values(df: pd.DataFrame, relevant_features: list, ncar: str) -> list:
    '''
    The function takes a pandas DataFrame as input and checks for the existence of any NaN values. It returns a list of columns that contain NaN values. 
    Args: 
        df: A pandas DataFrame 
    Return: 
        columns_with_nan: A list of columns that contain NaN values in the input DataFrame. If no NaN values are present, an empty list is returned.
    '''
    df = df[relevant_features]
    columns_with_nan = df.columns[df.isna().any()].tolist()
    if len(columns_with_nan) > 0:
        logger.error(f"{ncar}: There are car parts in the dataset with NaN values in the following columns: {columns_with_nan}")
    
    return columns_with_nan

def combine_dataframes(dataframes: list, relevant_features: list, ncars: list) -> pd.DataFrame:
    '''
    The function takes a list of pandas DataFrames and combines them into a single data frame. Before merging, it checks if all dataframes have the same columns and returns an error if there are discrepancies. 
    If any NaN values exist in the input data frames, it uses the check_nan_values function to obtain the list of columns with the NaN values. 
    It returns a single merged dataframe containing all columns from all input data frames. 
    Args: 
        dataframes: A list of pandas DataFrame objects. 
    Return: 
        merged_df: A single pandas DataFrame object that contains all rows and columns from all input data frames
    '''
    # Set the header information
    logger.info("Combine all datasets to one...")
    columns_set = set(dataframes[0].columns)
    # Check if all dataframes have the same columns 
    for df, ncar in zip(dataframes, ncars):
        cols_with_nan_values = check_nan_values(df=df, relevant_features=relevant_features, ncar=ncar)
        if set(df.columns) != columns_set:
            logger.info(df.columns)
            logger.info(columns_set)
            raise ValueError("All dataframes must have the same columns.")
    
    # Merge all dataframes into a single dataframe
    merged_df = pd.concat(dataframes).reset_index(drop=True)
    
    logger.success(f"{len(dataframes)} dataframe(s) are combined to one dataset.")
    
    return merged_df    

def store_trained_model(model, metrics: str, best_iteration: int, val_auc: float, hp: dict, index_best_model: int, model_folder_path: str, finalmodel: bool) -> None:
    ''' 
    This function stores the trained model, hyperparameters, metrics and best iteration information in a pickled file at the provided model folder path and logs the validation AUC and training information in a txt file.
    Args:
        model: The trained model.
        metrics: A dictionary containing the evaluation metrics and their values for the model.
        best_iteration: An integer indicating the number of iterations the model took to converge to the best solution.
        val_auc: A float representing the AUC score of the validation set.
        hp: A dictionary containing the hyperparameters used to train the model.
        index_best_model: An integer indicating the index of the best model in the hyperparameter tuning process.
        model_folder_path: A string indicating the path where the trained model and logs will be saved.
        finalmodel: A boolean value indicating whether the model to be saved is the final model.
    Return: None
    '''
    # save model
    if finalmodel:
            model_path = model_folder_path + f"final_model.pkl"
    else:
            model_path = model_folder_path + f"model.pkl"

    with open(model_path, "wb") as filestore:
        pickle.dump(model, filestore)

    logging_file_path = model_folder_path + "logging.txt"
    if os.path.isfile(logging_file_path):
        log_text = "Validation AUC (final model): {}\n".format(val_auc)
        f= open(model_folder_path + "logging.txt","a")
        f.write("\n_________________________________________________\n")
        f.write("Final model:\n")
        f.write(log_text)
        f.write("Trained Iterations: {}\n".format(best_iteration))
        f.close()
    else:
        dataset_path = "Dataset: {}\n".format(config["train_settings"]["folder_processed_dataset"])
        model_folder = "Model folder path: {}\n".format(model_folder_path)
        f= open(model_folder_path + "logging.txt","w+")
        f.write(dataset_path)
        f.write(model_folder)
        f.write("use_only_text: {}\n".format(config["dataset_params"]["use_only_text"]))
        f.write("Method: {}".format(config["train_settings"]["ml-method"]))
        f.write("\n_________________________________________________\n")
        f.write("Best model after hyperparameter tuning:\n")
        f.write("Validation AUC: {}\n".format(val_auc))
        f.write("Trained Iterations: {}\n".format(best_iteration))
        f.write("Model index in hyperparameter tuning: {}\n".format(index_best_model+1))
        f.write("Hyperparameter:\n")
        for key in hp:
            f.write("{}: {}\n".format(key, hp[key]))
        f.write(f"Metrics: {metrics} \n")
        f.write(json.dumps(config["train_settings"]))
        f.write("\n")
        f.write(json.dumps(config["prediction_settings"]))
        f.write("\n")
        f.close()