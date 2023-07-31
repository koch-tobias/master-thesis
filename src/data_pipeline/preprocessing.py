# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import os
import pickle
import shutil
from loguru import logger

from src.data_pipeline.feature_engineering import transform_boundingbox, calculate_center_point, calculate_lwh, calculate_orientation, clean_text, nchar_text_to_vec, doc2vec_text_to_vec, bert_text_to_vec
from src.config import general_params, convert_dict, train_settings

# %%
def load_csv_into_df(original_prisma_data: bool, label_new_data: bool) -> list:
    '''
    This function searches for all .xls files in a given directory, loads each file into a pandas dataframe and changes the header line.
    return: List with all created dataframes
    '''

    # Check if the folder exists
    if label_new_data:
        folder_name = "data/raw_for_labeling"
    else:
        folder_name = "data/labeled"
        
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
                    # Load the excel into a pandas dataframe, delete the header and declare the second row as new header
                    if original_prisma_data == True:
                        df = pd.read_excel(os.path.join(folder_name, file), header=None, skiprows=1)
                        df.columns = df.iloc[0]
                        df = df.iloc[1:]
                        ncar = df[general_params["car_part_designation"]][1].split(" ")[0]
                    else:
                        df = pd.read_csv(os.path.join(folder_name, file))
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
def check_nan_values(df):
    columns = ['X-Min','X-Max','Y-Min','Y-Max','Z-Min','Z-Max', 'ox','oy', 'oz', 'xx','xy','xz', 'yx','yy','yz','zx','zy','zz']
    nan_rows = df[df[columns].isnull().any(axis=1)]
    if not nan_rows.empty:
        print("Rows with NaN values in columns:", columns)
        print(nan_rows)
        logger.error(f"Please check your data. There are car parts in the dataset with nan values in the following columns: {columns}")

# %%
def combine_dataframes(dataframes: list) -> pd.DataFrame:
    '''
    This function takes a list of data frames as input and checks if the dataframes have the same header. If so, the dataframes will be merged.
    return: Merged dataframe
    '''
    # Set the header information

    logger.info("Combine all datasets to one...")
    columns_set = set(dataframes[0].columns)
    # Check if all dataframes have the same columns 
    for df in dataframes:
        check_nan_values(df)
        if set(df.columns) != columns_set:
            logger.info(df.columns)
            logger.info(columns_set)
            raise ValueError("All dataframes must have the same columns.")
    
    # Merge all dataframes into a single dataframe
    merged_df = pd.concat(dataframes).reset_index(drop=True)
    
    logger.success(f"{len(dataframes)} dataframe(s) are combined to one dataset.")
    
    return merged_df    

# %%
def prepare_and_add_labels(dataframe: pd.DataFrame):

    logger.info("Start preprocessing the data...")

    # Store the ncar abbreviation for file paths
    ncar = dataframe[general_params["car_part_designation"]][1].split(" ")[0]

    # Keep only car parts of module group EF
    index_EF_module = dataframe[dataframe[general_params["car_part_designation"]].str.contains('EF')].index[-1]
    dataframe_new = dataframe.loc[:index_EF_module-1]

    for modules in general_params["keep_modules"]:
        try:
            level = dataframe[dataframe[general_params["car_part_designation"]].str.contains(modules)]["Ebene"].values[0]
            startindex = dataframe[dataframe[general_params["car_part_designation"]].str.contains(modules)].index[-1]+1
            endindex = dataframe.loc[(dataframe["Ebene"] == level) & (dataframe.index > startindex)].index[0]-1
            temp = dataframe.loc[startindex:endindex]
            dataframe_new = pd.concat([dataframe_new, temp], ignore_index=True).reset_index(drop=True)
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

    dataframe_new = dataframe_new.reset_index(drop=True)

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
    df_relevants = pd.concat([df_relevants, df_temp], ignore_index=True).reset_index(drop=True)

    df_relevants = clean_text(df_relevants)

    # Drop the mirrored car parts (on the right sight) which have the same Sachnummer
    df_new = df_relevants.drop_duplicates(subset='Sachnummer', keep=False)
    df_filtered = df_relevants[df_relevants.duplicated(subset='Sachnummer', keep=False)]
    df_filtered = df_filtered[df_filtered['yy'].astype(float) >= 0]
    df_relevants = pd.concat([df_new, df_filtered], ignore_index=True).reset_index(drop=True)

    # Drop the mirrored car parts (on the right sight) which have not the same Sachnummer 
    df_relevants = df_relevants.loc[~(df_relevants.duplicated(subset='Kurzname', keep=False) & (df_relevants['L/R-Kz.'] == 'R'))]

    df_for_plot = df_relevants[(df_relevants['X-Min'] != 0) & (df_relevants['X-Max'] != 0)]

    # Reset the index of the merged data frame
    df_relevants = df_relevants.reset_index(drop=True)

    df_relevants = df_relevants.drop_duplicates().reset_index(drop=True)

    logger.success(f"The dataset is successfully preprocessed. The new dataset contains {df_relevants.shape[0]} samples")

    return df_relevants, df_for_plot

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

# %%
def train_test_val(df, model_folder_path, binary_model):
    if binary_model:
        logger.info("Split the dataset into train validation and test sets for the binary task and store the sets in dictionaries...")
    else:
        logger.info("Split the dataset into train validation and test sets for the multiclass task and store the sets in dictionaries......")
    
    X = nchar_text_to_vec(df, model_folder_path) # Using ngram vectorizer
    #X = doc2vec_text_to_vec(df, model_folder_path)
    #X = bert_text_to_vec(df, model_folder_path)

    # Combine text features with other features
    features = general_params["features_for_model"]
    bbox_features_dict = {"features_for_model": features}
    with open(model_folder_path + 'boundingbox_features.pkl', 'wb') as fp:
        pickle.dump(bbox_features_dict, fp)

    if general_params["use_only_text"] == False:
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

    X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(X, y, indices, test_size=train_settings["train_val_split"], stratify=y, random_state=general_params["seed"])
    X_val, X_test, y_val, y_test, indices_val, indices_test = train_test_split(X_val, y_val, indices_val, test_size=train_settings["val_test_split"], stratify=y_val, random_state=general_params["seed"])

    df_train = df.iloc[indices_train]
    df_val = df.iloc[indices_val]
    df_test = df.iloc[indices_test]

    logger.success("Train, validation and test sets are generated!")

    return X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test, weight_factor