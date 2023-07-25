# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import shutil
from loguru import logger
import random
import pickle

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from data.augmentation import data_augmentation
from data.text_preperation import vectorize_data, get_vocabulary, clean_text
from visualization.plot_functions import analyse_data
from data.boundingbox_calculations import transform_boundingbox, calculate_center_point, calculate_lwh, calculate_orientation
from config_model import general_params, paths, train_settings, convert_dict

# %%
def load_csv_into_df(original_prisma_data: bool, label_new_data: bool) -> list:
    '''
    This function searches for all .xls files in a given directory, loads each file into a pandas dataframe and changes the header line.
    return: List with all created dataframes
    '''

    # Check if the folder exists
    if label_new_data:
        folder_name = paths["new_data"]
    else:
        folder_name = paths["labeled_data"]
        
    if not os.path.exists(folder_name):
        logger.error(f"The path {folder_name} does not exist.")
        exit()
    else:
        logger.info("Loading the data...")

        # Create an empty list to store all dataframes
        dataframes = []
        ncars = []
        # Loop through all files in the folder and open them as dataframes
        for file in os.listdir(folder_name):
            if file.endswith(".xls") or file.endswith(".xlsx"):
                try:
                    # Load the excel into a pandas dataframe, delete the header and declare the second row as new header
                    if original_prisma_data == True:
                        df = pd.read_excel(os.path.join(folder_name, file), header=None, skiprows=1)
                        df.columns = df.iloc[0]
                        df = df.iloc[1:]
                        ncar = df[general_params["car_part_designation"]][1].split(" ")[0]
                    else:
                        df = pd.read_excel(os.path.join(folder_name, file))
                        ncar = file.split("_")[0]
                        df["Derivat"] = ncar
                    # Add the created dataframe to the list of dataframes
                    dataframes.append(df)
                    ncars.append(ncar)

                    if original_prisma_data == True:
                        old_path = os.path.join(folder_name, file)
                        new_path = os.path.join("data/raw", ncar + '_' + file) 
                        shutil.move(old_path, new_path)

                except:
                    logger.info(f"Error reading file {file}. Skipping...")
                    continue

    # Check if any dataframes were created
    if len(dataframes) == 0:
        logger.error(f"No dataframes were created - please check if the files in folder {folder_name} are correct/exist.")
        exit()
    else:
        logger.success(f"{len(dataframes)} dataframe(s) were created.")

        return dataframes, ncars

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
            logger.info(df.columns)
            logger.info(columns_set)
            raise ValueError("All dataframes must have the same columns.")
    
    dataframes = dataframes[:5]
    # Merge all dataframes into a single dataframe
    merged_df = pd.concat(dataframes, ignore_index=True)

    logger.success(f"{len(dataframes)} dataframe(s) are combined to one dataset.")
    
    return merged_df    

# %%
def prepare_and_add_labels(dataframe: pd.DataFrame):

    logger.info("Start preprocessing the data...")

    # Store the ncar abbreviation for file paths
    ncar = dataframe[general_params["car_part_designation"]][1].split(" ")[0]

    # Keep only car parts of module group EF
    index_EF_module = dataframe[dataframe[general_params["car_part_designation"]].str.contains('EF')].index[-1]
    #index_EF_module = dataframe[dataframe[general_params["car_part_designation"]].str.startswith(f'EF {ncar}')].index[-1]
    dataframe_new = dataframe.loc[:index_EF_module-1]

    for modules in general_params["keep_modules"]:
        try:
            level = dataframe[dataframe[general_params["car_part_designation"]].str.contains(modules)]["Ebene"].values[0]
            startindex = dataframe[dataframe[general_params["car_part_designation"]].str.contains(modules)].index[-1]+1
            endindex = dataframe.loc[(dataframe["Ebene"] == level) & (dataframe.index > startindex)].index[0]-1
            temp = dataframe.loc[startindex:endindex]
            dataframe_new = pd.concat([dataframe_new, temp]).reset_index(drop=True)
        except:
            logger.info(f"Module {modules} in structure tree not found!")

    # Keep only the relevant samples with Dok-Format=5P. This samples are on the last level of the car structure
    dataframe_new = dataframe_new[dataframe_new["Dok-Format"]=='5P'].reset_index(drop=True)

    # Delete the NCAR abbreviation because of data security reasons
    dataframe_new[general_params["car_part_designation"]] = dataframe_new[general_params["car_part_designation"]].apply(lambda x: x.replace(ncar, ""))

    # Keep only features which are identified as relevant for the preprocessing, the predictions or for the users' next steps
    dataframe_new = dataframe_new[general_params["relevant_features"]]
    
    dataframe_new = dataframe_new.astype(convert_dict)

    # Add columns for the label "Relevant für Messung" and "Einheitsname"
    dataframe_new.insert(len(dataframe_new.columns), 'Relevant fuer Messung', 'Nein')
    dataframe_new.insert(len(dataframe_new.columns), 'Einheitsname', 'Dummy')

    logger.success(f"The features are reduced and formated to the correct data type!")
    
    return dataframe_new, ncar

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
def preprocess_dataset(df):
    logger.info(f"Start preprocessing the dataframe with {df.shape[0]} samples...")

    df.loc[df['X-Max'] == 10000, ['X-Min', 'X-Max', 'Y-Min', 'Y-Max', 'Z-Min', 'Z-Max', 'ox', 'oy', 'oz', 'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz', 'Wert']] = 0

    df_new_features = add_new_features(df)

    ##########
    # OUTLIER DETECTION
    # Calculate the upper and lower limits
    Q1 = df_new_features['X-Max_transf'].quantile(0.25)
    Q3 = df_new_features['X-Max_transf'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    
    # Create arrays of Boolean values indicating the outlier rows
    upper_array = np.where(df_new_features['X-Max_transf']>=upper)[0]
    lower_array = np.where(df_new_features['X-Max_transf']<=lower)[0]
    
    # Set the bounding box features to zero if detected as outlier
    df_new_features.loc[upper_array, general_params["bounding_box_features_original"]] = 0
    df_new_features.loc[lower_array, general_params["bounding_box_features_original"]] = 0
    #################

    # Using dictionary to convert specific columns
    df_new_features = df_new_features.astype(convert_dict)

    df_new_features = df_new_features[(df_new_features['X-Min'] != 0) & (df_new_features['X-Max'] != 0)]

    # Save the samples without/wrong bounding box information in a new df, as they will need to be added back later
    df_temp = df[(df["X-Min"] == 0.0) & (df["X-Max"] == 0.0)]

    # Delete all samples which have less volume than 500,000 mm^3
    df_relevants = df_new_features[(df_new_features['volume'] > 500000)].reset_index(drop=True)

    # Delete all samples where the parts are in the front area of the car
    x_min_transf, x_max_transf = df_relevants["X-Min_transf"].min(), df_relevants["X-Max_transf"].max()
    car_length = x_max_transf - x_min_transf
    cut_point_x = x_min_transf + car_length*general_params["cut_percent_of_front"]
    df_relevants = df_relevants[df_relevants["X-Min_transf"] > cut_point_x]
    # Concatenate the two data frames vertically
    df_relevants = pd.concat([df_relevants, df_temp]).reset_index(drop=True)

    df_relevants = clean_text(df_relevants)

    # Drop the mirrored car parts (on the right sight) which have the same Sachnummer
    df_new = df_relevants.drop_duplicates(subset='Sachnummer', keep=False)
    df_filtered = df_relevants[df_relevants.duplicated(subset='Sachnummer', keep=False)]
    df_filtered = df_filtered[df_filtered['yy'].astype(float) >= 0]
    df_relevants = pd.concat([df_new, df_filtered]).reset_index(drop=True)

    # Drop the mirrored car parts (on the right sight) which have not the same Sachnummer 
    df_relevants = df_relevants.loc[~(df_relevants.duplicated(subset='Kurzname', keep=False) & (df_relevants['L/R-Kz.'] == 'R'))]

    df_for_plot = df_relevants[(df_relevants['X-Min'] != 0) & (df_relevants['X-Max'] != 0)]

    # Reset the index of the merged data frame
    df_relevants = df_relevants.reset_index(drop=True)

    df_relevants = df_relevants.drop_duplicates()

    logger.success(f"The dataset is successfully preprocessed. The new dataset contains {df_relevants.shape[0]} samples")

    return df_relevants, df_for_plot

# %%
def train_test_val(df, model_folder_path, binary_model):
    logger.info("Generate train, validation and test split...")
    X = vectorize_data(df, model_folder_path)

    # Combine text features with other features
    features = general_params["features_for_model"]
    bbox_features_dict = {"features_for_model": features}
    with open(model_folder_path + 'boundingbox_features.pkl', 'wb') as fp:
        pickle.dump(bbox_features_dict, fp)

    if train_settings["use_only_text"] == False:
        X = np.concatenate((X, df[features].values), axis=1)

    if binary_model:
        y = df['Relevant fuer Messung']
        y = y.map({'Ja': 1, 'Nein': 0})
    else:
        y = df['Einheitsname']
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        with open(model_folder_path + 'label_encoder.pkl', 'wb') as f: 
            pickle.dump(le, f)  

    weight_factor = get_weight_factor(y, df, binary_model)     

    indices = np.arange(X.shape[0])

    X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(X, y, indices, test_size=train_settings["val_size"], stratify=y, random_state=42)
    X_val, X_test, y_val, y_test, indices_val, indices_test = train_test_split(X_val, y_val, indices_val, test_size=train_settings["test_size"], stratify=y_val, random_state=42)

    df_test = df.iloc[indices_test]

    logger.success("Train, validation and test sets are generated!")

    return X_train, y_train, X_val, y_val, X_test, y_test, df_test, weight_factor

# %%
def get_weight_factor(y, df, binary_model):
    if binary_model == False:
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

def get_model(folder_path):
    final_model_path = ""
    pre_model_path = ""
    # Load model for relevance
    for file in os.listdir(folder_path):
        if file.startswith("final"):
            final_model_path =  os.path.join(folder_path, file)
        elif file.startswith("model"):
            pre_model_path =  os.path.join(folder_path, file)

    if final_model_path != "":
        model_path = final_model_path
    else:
        model_path = pre_model_path

    with open(model_path, "rb") as fid:
        lgbm = pickle.load(fid)

    # Load the vectorizer from the file
    vectorizer_path = folder_path + "/vectorizer.pkl"
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    # Get the vocabulary of the training data
    vocab_path = folder_path + "/vocabulary.pkl"
    with open(vocab_path, 'rb') as f:
        vocabulary = pickle.load(f) 

    bbox_features_path = folder_path + "/boundingbox_features.pkl"
    with open(bbox_features_path, 'rb') as f:
        bbox_features = pickle.load(f)  

    return lgbm, vectorizer, vocabulary, bbox_features

def get_X(vocab, vectorizer, bbox_features, df_preprocessed):
    # Convert the vocabulary list to a dictionary
    vocabulary_dict = {word: index for index, word in enumerate(vocab)}

    # Set the vocabulary of the vectorizer to the loaded vocabulary
    vectorizer.vocabulary_ = vocabulary_dict
    X = vectorizer.transform(df_preprocessed['Benennung (bereinigt)']).toarray()

    # Combine text features with other features
    if train_settings["use_only_text"] == False:      
        X = np.concatenate((X, df_preprocessed[bbox_features].values), axis=1)
    
    return X

# %%
def load_prepare_dataset(model_folder_path, binary_model: bool):

    path_trainset = "models/df_trainset.xlsx"

    if os.path.exists(path_trainset):
        df_preprocessed = pd.read_excel(path_trainset) 
    else:
        dataframes_list, ncars = load_csv_into_df(original_prisma_data=False, label_new_data=False)

        df_combined = combine_dataframes(dataframes_list)
        df_preprocessed, df_for_plot = preprocess_dataset(df_combined)

        if train_settings["augmentation"]:
            # Generate the new dataset
            df_preprocessed = data_augmentation(df_preprocessed)

        df_preprocessed.to_excel("models/df_trainset.xlsx")

    X_train, y_train, X_val, y_val, X_test, y_test, df_test, weight_factor = train_test_val(df_preprocessed, model_folder_path=model_folder_path, binary_model=binary_model)

    analyse_data(df_preprocessed, y_train, y_val, y_test, model_folder_path, binary_model)

    return X_train, y_train, X_val, y_val, X_test, y_test, df_preprocessed, df_test, weight_factor
