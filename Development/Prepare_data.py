# %%
import pandas as pd
import os
from loguru import logger
from pathlib import Path
from datetime import datetime
import shutil
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import random
from text_preprocessing import vectorize_data
from Feature_Engineering import preprocess_dataset
from text_preprocessing import get_vocabulary
from Data_Augmentation import data_augmentation
from config import general_params, convert_dict, paths, train_settings

# %% [markdown]
# ### Functions

# %%
def load_csv_into_df(original_prisma_data: bool) -> list:
    '''
    This function searches for all .xls files in a given directory, loads each file into a Pandas dataframe and changes the header line.
    If move_to_archive is set True, then all processed files will be moved to the archive.
    return: List with all created dataframes
    '''
    # Check if the folder exists
    folder_name = paths["labeled_data"]
    if not os.path.exists(folder_name):
        logger.error(f"The path {folder_name} does not exist.")
        exit()
    else:
        logger.info("Loading the data...")

        # Create an empty list to store all dataframes
        dataframes = []
        
        # Loop through all files in the folder and open them as dataframes
        for file in os.listdir(folder_name):
            if file.endswith(".xls") or file.endswith(".xlsx"):
                try:
                    # Load the excel into a pandas dataframe, delete the header and declare the second row as new header
                    if original_prisma_data == True:
                        df = pd.read_excel(os.path.join(folder_name, file), header=None, skiprows=1)
                        df.columns = df.iloc[0]
                        df = df.iloc[1:]
                    else:
                        df = pd.read_excel(os.path.join(folder_name, file))

                    # Add the created dataframe to the list of dataframes
                    dataframes.append(df)

                except:
                    logger.info(f"Error reading file {file}. Skipping...")
                    continue

    # Check if any dataframes were created
    if len(dataframes) == 0:
        logger.error(f"No dataframes were created - please check if the files in folder {folder_name} are correct/exist.")
        exit()
    else:
        logger.success(f"{len(dataframes)} dataframe(s) were created.")

        return dataframes

# %%
def combine_dataframes(dataframes: list) -> pd.DataFrame:
    '''
    This function takes a list of data frames as input and checks if the dataframes have the same header. If so, the dataframes will be merged.
    return: Merged dataframe
    '''
    # Set the header information
    columns_set = set(dataframes[0].columns)

    # Check if all dataframes have the same columns 
    for df in dataframes:
        if set(df.columns) != columns_set:
            print(df.columns)
            print(columns_set)
            raise ValueError("All dataframes must have the same columns.")
    
    # Merge all dataframes into a single dataframe
    merged_df = pd.concat(dataframes, ignore_index=True)

    logger.success(f"{len(dataframes)} dataframe(s) are combined to one dataset.")
    
    return merged_df    

# %%
def prepare_and_add_labels(dataframes: list):

    logger.info("Start preprocessing the data...")
    dataframes_with_labels = []
    ncars = []

    for i in range(len(dataframes)):
        # Store the ncar abbreviation for file paths
        ncar = dataframes[i][general_params["car_part_designation"]][1][:3]
        ncars.append(ncar)

        for modules in general_params["keep_modules"]:
            # Temporary store the module for the interior mirror
            level = dataframes[i][dataframes[i][general_params["car_part_designation"]].str.startswith(f'{ncar} {modules}')]["Ebene"].values[0]
            startindex = dataframes[i][dataframes[i][general_params["car_part_designation"]].str.startswith(f'{ncar} {modules}')].index[-1]+1
            endindex = dataframes[i].loc[(dataframes[i]["Ebene"] == level) & (dataframes[i].index > startindex)].index[0]-1
            temp = dataframes[i].loc[startindex:endindex]
            dataframes[i] = pd.concat([dataframes[i], temp]).reset_index(drop=True)

        # Temporary store the module for the roof antenna
        level_roof_antenna = dataframes[i][dataframes[i][general_params["car_part_designation"]].str.startswith(f'{ncar} CD07')]["Ebene"].values[0]
        startindex_roof_antenna = dataframes[i][dataframes[i][general_params["car_part_designation"]].str.startswith(f'{ncar} CD07')].index[-1]+1
        endindex_roof_antenna = dataframes[i].loc[(dataframes[i]["Ebene"] == level_roof_antenna) & (dataframes[i].index > startindex_roof_antenna)].index[0]-1
        temp_roof_antenna = dataframes[i].loc[startindex_roof_antenna:endindex_roof_antenna]

        # Keep only car parts of module group EF
        index_EF_module = dataframes[i][dataframes[i][general_params["car_part_designation"]].str.startswith(f'EF {ncar}')].index[-1]
        dataframes[i] = dataframes[i].loc[:index_EF_module-1]

        # Keep only the relevant samples with Dok-Format=5P. This samples are on the last level of the car structure
        dataframes[i] = dataframes[i][dataframes[i]["Dok-Format"]=='5P'].reset_index(drop=True)

        # Delete the NCAR abbreviation because of data security reasons
        dataframes[i][general_params["car_part_designation"]] = dataframes[i][general_params["car_part_designation"]].apply(lambda x: x.replace(ncar, ""))

        # Keep only features which are identified as relevant for the preprocessing, the predictions or for the users' next steps
        dataframes[i] = dataframes[i][general_params["relevant_features"]]
       
        dataframes[i] = dataframes[i].astype(convert_dict)

        # Add columns for the label "Relevant für Messung" and "Allgemeine Bezeichnung"
        data_labeled = dataframes[i]
        data_labeled.insert(len(data_labeled.columns), 'Relevant fuer Messung', 'Nein')
        data_labeled.insert(len(data_labeled.columns), 'Einheitsname', 'Dummy')
        dataframes_with_labels.append(data_labeled)

        if general_params["save_prepared_dataset_for_labeling"]:
            # Date
            dateTimeObj = datetime.now()
            timestamp = dateTimeObj.strftime("%d%m%Y_%H%M")
            
            # Store preprocessed dataframes
            dataframes_with_labels[i].to_excel(f"../data/preprocessed_data/{ncar}_preprocessed_{timestamp}.xlsx")

            logger.success(f"The features are reduced and formated to the correct data type. The new dataset is stored as {ncar}_preprocessed_{timestamp}.xlsx!")
        else:
            logger.success(f"The features are reduced and formated to the correct data type!")
    
    return dataframes_with_labels, ncars

# %%
def train_test_val(df, df_test, test_size:float, timestamp):
    
    X, X_test = vectorize_data(df, df_test, timestamp)

    # Combine text features with other features
    features = general_params["features_for_model"]
    if train_settings["use_only_text"] == False:
        X = np.concatenate((X, df[features].values), axis=1)
        X_test = np.concatenate((X_test, df_test[features].values), axis=1)

    if train_settings["classify_einheitsnamen"] == False:
        y = df['Relevant fuer Messung']
        y = y.map({'Ja': 1, 'Nein': 0})
        y_test = df_test['Relevant fuer Messung']
        y_test = y_test.map({'Ja': 1, 'Nein': 0})
    else:
        y = df['Einheitsnamen']
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        y_test = df_test['Einheitsnamen']
        y_test = le.transform(y_test)     

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test

# %%
def train_test_val_kfold(df, df_test, timestamp):

    X, X_test = vectorize_data(df, df_test, timestamp)

    # Combine text features with other features
    if train_settings["use_only_text"] == False:
        features = general_params["features_for_model"]
        X = np.concatenate((X, df[features].values), axis=1)

    if train_settings["classify_einheitsnamen"] == False:
        y = df['Relevant fuer Messung']
        y = y.map({'Ja': 1, 'Nein': 0})
        y_test = df_test['Relevant fuer Messung']
        y_test = y_test.map({'Ja': 1, 'Nein': 0})
    else:
        y = df['Einheitsnamen']
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        y_test = df_test['Einheitsnamen']
        y_test = le.transform(y_test)     

    return X, y, X_test, y_test

# %%
def load_prepare_dataset(test_size):
    dataframes_list = load_csv_into_df(original_prisma_data=False)
    random.seed(33)
    # Take random dataset from list as test set and drop it from the list
    random_index = random.randint(0, len(dataframes_list) - 1)
    ncar = general_params["ncars"]
    df_test = dataframes_list[random_index]
    dataframes_list.pop(random_index)
    logger.info(f"Car {ncar[random_index]} is used to test the model on unseen data!")
    df_combined = combine_dataframes(dataframes_list)
    df_preprocessed, df_for_plot = preprocess_dataset(df_combined, cut_percent_of_front=general_params["cut_percent_of_front"])
    df_test, df_test_for_plot = preprocess_dataset(df_test, cut_percent_of_front=general_params["cut_percent_of_front"])

    if general_params["save_preprocessed_data"]:
        df_preprocessed.to_excel("df_preprocessed.xlsx")
        df_test.to_excel("df_test.xlsx")

    vocab = get_vocabulary(df_preprocessed['Benennung (bereinigt)'])

    if train_settings["augmentation"]:
        # Generate the new dataset
        df_preprocessed = data_augmentation(df_preprocessed)

    weight_factor = round(df_preprocessed[df_preprocessed["Relevant fuer Messung"]=="Nein"].shape[0] / df_preprocessed[df_preprocessed["Relevant fuer Messung"]=="Ja"].shape[0])

    dateTimeObj = datetime.now()
    timestamp = dateTimeObj.strftime("%d%m%Y_%H%M")

    # Split dataset
    if train_settings["cross_validation"]:
        X, y, X_test, y_test = train_test_val_kfold(df_preprocessed, df_test, test_size=test_size, timestamp=timestamp)
        return X, y, X_test, y_test, weight_factor, timestamp, vocab
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = train_test_val(df_preprocessed, df_test, test_size=test_size, timestamp=timestamp)
        return X_train, y_train, X_val, y_val, X_test, y_test, weight_factor, timestamp, vocab

# %% [markdown]
# ### Main

# %%
def main():
    # Define the path to the folder containing the data (xls files)
    dataframes = load_csv_into_df(original_prisma_data=False)
    df = combine_dataframes(dataframes)


# %%
if __name__ == "__main__":
    
    main()


