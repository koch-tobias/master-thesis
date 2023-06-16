# %%
import pandas as pd
import numpy as np

import os
from loguru import logger
from pathlib import Path
from datetime import datetime
import shutil
import random
import pickle

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from Data_Augmentation import data_augmentation
from text_preprocessing import vectorize_data, get_vocabulary, clean_text
from boundingbox_calculations import transform_boundingbox, calculate_center_point, calculate_lwh, calculate_orientation
from plot_functions import plot_vehicle
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
def add_new_features(df):
    for index, row in df.iterrows():  
        # Calculate and add new features to represent the bounding boxes
        transformed_boundingbox = transform_boundingbox(row['X-Min'], row['X-Max'], row['Y-Min'], row['Y-Max'], row['Z-Min'], row['Z-Max'],row['ox'],row['oy'],row['oz'],row['xx'],row['xy'],row['xz'],row['yx'],row['yy'],row['yz'],row['zx'],row['zy'],row['zz'])
        center_x, center_y, center_z = calculate_center_point(transformed_boundingbox)
        length, width, height = calculate_lwh(transformed_boundingbox)
        theta_x, theta_y, theta_z = calculate_orientation(transformed_boundingbox)

        x_coords = transformed_boundingbox[:, 0]
        y_coords = transformed_boundingbox[:, 1]
        z_coords = transformed_boundingbox[:, 2]

        df.at[index, 'X-Min_transf'] = min(x_coords)
        df.at[index, 'X-Max_transf'] = max(x_coords)
        df.at[index, 'Y-Min_transf'] = min(y_coords)
        df.at[index, 'Y-Max_transf'] = max(y_coords)
        df.at[index, 'Z-Min_transf'] = min(z_coords)
        df.at[index, 'Z-Max_transf'] = max(z_coords)   
        df.at[index, 'center_x'] = center_x
        df.at[index, 'center_y'] = center_y
        df.at[index, 'center_z'] = center_z
        df.at[index, 'length'] = length
        df.at[index, 'width'] = width
        df.at[index, 'height'] = height
        df.at[index, 'theta_x'] = theta_x
        df.at[index, 'theta_y'] = theta_y
        df.at[index, 'theta_z'] = theta_z

        # Calculate and add the volume as new feature 
        volume = length * width * height
        df.at[index, 'volume'] = volume

        # If weight is availabe, calculate and add the density as new feature 
        if pd.notnull(row['Wert']) and volume != 0:
            density = row['Wert'] / volume
            df.at[index, 'density'] = density
        
    df.loc[df['Wert'].isnull(), ['Wert']] = 0
    df.loc[df['density'].isnull(), ['density']] = 0
        
    return df

# %%
def preprocess_dataset(df, cut_percent_of_front: float):
    logger.info(f"Start preprocessing the dataframe with {df.shape[0]} samples...")

    df.loc[df['X-Max'] == 10000, ['X-Min', 'X-Max', 'Y-Min', 'Y-Max', 'Z-Min', 'Z-Max', 'ox', 'oy', 'oz', 'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz', 'Wert']] = 0

    df_new_features = add_new_features(df)

    # Using dictionary to convert specific columns
    convert_dict = {'X-Min': float,
                    'X-Max': float,
                    'Y-Min': float,
                    'Y-Max': float,
                    'Z-Min': float,
                    'Z-Max': float,
                    'Wert': float,
                    'ox': float,
                    'oy': float,
                    'oz': float,
                    'xx': float,
                    'xy': float,
                    'xz': float,
                    'yx': float,
                    'yy': float,
                    'yz': float,
                    'zx': float,
                    'zy': float,
                    'zz': float                     
                    }
    
    df_new_features = df_new_features.astype(convert_dict)

    df_new_features = df_new_features[(df_new_features['X-Min'] != 0) & (df_new_features['X-Max'] != 0)]

    # Save the samples without/wrong bounding box information in a new df, as they will need to be added back later
    df_temp = df[(df["X-Min"] == 0.0) & (df["X-Max"] == 0.0)]

    # Delete all samples which have less volume than 500,000 mm^3
    df_relevants = df_new_features[(df_new_features['volume'] > 500000)].reset_index(drop=True)

    # Delete all samples where the parts are in the front area of the car
    x_min_transf, x_max_transf = df_relevants["X-Min_transf"].min(), df_relevants["X-Max_transf"].max()
    car_length = x_max_transf - x_min_transf
    cut_point_x = x_min_transf + car_length*cut_percent_of_front
    df_relevants = df_relevants[df_relevants["X-Min_transf"] > cut_point_x]

    # Concatenate the two data frames vertically
    df_relevants = pd.concat([df_relevants, df_temp]).reset_index(drop=True)

    df_relevants = clean_text(df_relevants)

    # Drop the mirrored car parts (on the right sight) which have the same Sachnummer
    df_relevants = df_relevants[~((df_relevants['Sachnummer'].duplicated(keep='last')) & (df_relevants['yy'] == -1))]

    # Drop the mirrored car parts (on the right sight) which have not the same Sachnummer 
    df_relevants = df_relevants.loc[~(df_relevants.duplicated(subset='Kurzname', keep=False) & (df_relevants['L/R-Kz.'] == 'R'))]

    df_for_plot = df_relevants[(df_relevants['X-Min'] != 0) & (df_relevants['X-Max'] != 0)]

    # Reset the index of the merged data frame
    df_relevants = df_relevants.reset_index(drop=True)

    logger.success(f"The dataset is successfully preprocessed. The new dataset contains {df_relevants.shape[0]} samples")

    return df_relevants, df_for_plot

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
        y = df['Einheitsname']
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        y_test = df_test['Einheitsname']
        y_test = le.transform(y_test)

        with open(f'../models/Einheitsnamen/lgbm_{timestamp}/label_encoder.pkl', 'wb') as f: 
            pickle.dump(le, f)  

    weight_factor = get_weight_factor(y, df)     

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test, weight_factor

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
        y = df['Einheitsname']
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        y_test = df_test['Einheitsname']
        y_test = le.transform(y_test)

        with open(f'../models/Einheitsnamen/lgbm_{timestamp}/label_encoder.pkl', 'wb') as f: 
            pickle.dump(le, f)

    weight_factor = get_weight_factor(y, df)     

    return X, y, X_test, y_test, weight_factor

# %%
def get_weight_factor(y, df):
    if train_settings["classify_einheitsnamen"]:
        # Get list of unique values in column "Einheitsname"

        unique_einheitsnamen = np.unique(y)
        weight_factor = {}
        for name in unique_einheitsnamen:
            weight_factor[name] = round(np.count_nonzero(y != name) / np.count_nonzero(y == name))
            if weight_factor[name] == 0:
                weight_factor[name] = 1
    else:    
        weight_factor = round(df[df["Relevant fuer Messung"]=="Nein"].shape[0] / df[df["Relevant fuer Messung"]=="Ja"].shape[0])

    return weight_factor


# %%
def load_prepare_dataset(test_size):
    if train_settings["already_preprocessed"] == False:
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

        if train_settings["augmentation"]:
            # Generate the new dataset
            df_preprocessed = data_augmentation(df_preprocessed)

        df_preprocessed.to_excel("df_preprocessed.xlsx")
        df_test.to_excel("df_test.xlsx")
    else:
        df_preprocessed = pd.read_excel("df_preprocessed.xlsx")
        df_test = pd.read_excel("df_test.xlsx")

    dateTimeObj = datetime.now()
    timestamp = dateTimeObj.strftime("%d%m%Y_%H%M")
    vocab = get_vocabulary(df_preprocessed['Benennung (bereinigt)'])

    # Split dataset
    if train_settings["cross_validation"]:
        X, y, X_test, y_test, weight_factor = train_test_val_kfold(df_preprocessed, df_test, test_size=test_size, timestamp=timestamp)
        return X, y, X_test, y_test, weight_factor, timestamp, vocab
    else:
        X_train, y_train, X_val, y_val, X_test, y_test, weight_factor = train_test_val(df_preprocessed, df_test, test_size=test_size, timestamp=timestamp)
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


