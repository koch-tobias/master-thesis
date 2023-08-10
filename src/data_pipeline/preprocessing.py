# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import os
import pickle
import shutil
from loguru import logger

from src.data_pipeline.feature_engineering import transform_boundingbox, calculate_center_point, calculate_lwh, calculate_orientation, clean_text, nchar_text_to_vec

import yaml
from yaml.loader import SafeLoader
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# %%
def load_data_into_df() -> tuple[list, str]:
    ''' 
    This function loads data from the specified folder path. It reads data from all files in the folder, converts them to pandas dataframes and stores the dataframes in a list. 
    The list of dataframes and a list of associated NCAR codes are returned as outputs. 
    Args:
        None
    Return:
        dataframes: a list containing pandas dataframes of all the files read from the specified folder path
        ncars: a list of the associated NCAR codes for all the files in the dataframes 
    '''

    # Check if the folder exists
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
                    df = pd.read_csv(os.path.join(folder_name, file))
                    ncar = file.split("_")[0]
                    df["Derivat"] = ncar

                    # Add the created dataframe to the list of dataframes
                    dataframes.append(df)
                    ncars.append(ncar)

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
def check_nan_values(df: pd.DataFrame, ncar: str) -> list:
    '''
    The function takes a pandas DataFrame as input and checks for the existence of any NaN values. It returns a list of columns that contain NaN values. 
    Args: 
        df: A pandas DataFrame 
    Return: 
        columns_with_nan: A list of columns that contain NaN values in the input DataFrame. If no NaN values are present, an empty list is returned.
    '''
    df = df[config["general_params"]["check_features_for_nan_values"]]
    columns_with_nan = df.columns[df.isna().any()].tolist()
    if len(columns_with_nan) > 0:
        logger.error(f"{ncar}: There are car parts in the dataset with NaN values in the following columns: {columns_with_nan}")
    return columns_with_nan

# %%
def combine_dataframes(dataframes: list, ncars: list) -> pd.DataFrame:
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
        cols_with_nan_values = check_nan_values(df, ncar)
        if set(df.columns) != columns_set:
            logger.info(df.columns)
            logger.info(columns_set)
            raise ValueError("All dataframes must have the same columns.")
    
    # Merge all dataframes into a single dataframe
    merged_df = pd.concat(dataframes).reset_index(drop=True)
    
    logger.success(f"{len(dataframes)} dataframe(s) are combined to one dataset.")
    
    return merged_df    

# %%
def check_if_columns_available(dataframe: pd.DataFrame, relevant_features: list) -> list:
    '''
    The function takes a pandas DataFrame and a list of relevant features/columns as input. 
    It checks if all relevant features/columns are present in the input DataFrame and returns a list of missing features/columns. 
    Args: 
        dataframe: A pandas DataFrame object 
        relevant_features: list of feature names that are required in the input DataFrame. 
    Return: 
        missing_columns: a list of features/columns that are missing in the input DataFrame. If all relevant features/columns are present in the input DataFrame, an empty list is returned.
    '''    
    missing_columns = []
    for column in relevant_features:
        if column not in dataframe.columns:
            missing_columns.append(column)
    
    return missing_columns

# %%
def prepare_and_add_labels(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    '''
    The function takes a pandas DataFrame as input and prepares the data by performing several data preprocessing steps. 
    It drops all empty columns, checks if all relevant features are available, stores the NCAR abbreviation for file paths, retains only the relevant samples with Dok-Format=5P and only keep relevant features. 
    It then creates and adds two new columns "Relevant fuer Messung" and "Einheitsname". Finally, it returns a tuple with the preprocessed DataFrame object and the NCAR abbreviation. 
    Args: 
        dataframe: A pandas DataFrame object. 
    Return: 
        dataframe: preprocessed pandas DataFrame object 
        ncar: string (NCAR abbreviation) which is used for file paths.
    '''
    logger.info("Start preparing the data...")

    # Drop all empty columns
    dataframe.dropna(how= "all", axis=1, inplace=True)

    missing_columns = check_if_columns_available(dataframe=dataframe, relevant_features=config["general_params"]["relevant_features"])
    if len(missing_columns) > 0:
        logger.exit(f"Please check your dataset. The following attributes are missing: {missing_columns}")

    # Store the ncar abbreviation for file paths
    ncar = dataframe['Code'].iloc[0]

    dataframe_new = pd.DataFrame(columns=dataframe.columns)

    for module in config["general_params"]["keep_modules"]:
        try: 
            for i in range(dataframe[dataframe["Modul (Nr)"] == module].shape[0]):
                level = dataframe[dataframe["Modul (Nr)"] == module]["Ebene"].values[i]
                startindex = dataframe[dataframe["Modul (Nr)"] == module].index[i]
                try:
                    endindex = dataframe.loc[(dataframe["Ebene"] == level) & (dataframe.index > startindex)].index[i] - 1
                except: 
                    endindex = dataframe.shape[0] + 1
                temp = dataframe.loc[startindex:endindex]
                dataframe_new = pd.concat([dataframe_new, temp], ignore_index=True).reset_index(drop=True)
        except:
            logger.info(f"Module {module} not found in the structure tree!")

    # Keep only the relevant samples with Dok-Format=5P. These are on the last level of the car structure and contains only car parts
    dataframe_new = dataframe_new[dataframe_new["Dok-Format"]=='5P'].reset_index(drop=True)

    # Delete the NCAR abbreviation due to data security
    dataframe_new[config["general_params"]["car_part_designation"]] = dataframe_new[config["general_params"]["car_part_designation"]].apply(lambda x: x.replace(ncar, ""))

    # Keep only features which are identified as relevant for the preprocessing, the predictions or for the users' next steps
    dataframe_new = dataframe_new[config["general_params"]["relevant_features"]]
    
    dataframe_new = dataframe_new.astype(config["convert_dict"])

    # Add columns for the label "Relevant für Messung" and "Einheitsname"
    dataframe_new.insert(len(dataframe_new.columns), 'Relevant fuer Messung', 'Nein')
    dataframe_new.insert(len(dataframe_new.columns), 'Einheitsname', 'Dummy')

    dataframe_new = dataframe_new.reset_index(drop=True)

    logger.success(f"Data ist prepared. The features are reduced and formated to the correct data type, subfolder are deleted, and only relevant modules are kept!")
    
    return dataframe_new, ncar

# %%
def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    The function takes a pandas DataFrame as input and adds new features/variables by calculating the bounding box coordinates, orientation, center point, length, width, height, volume, and density for each car part in the DataFrame. 
    It returns the updated pandas DataFrame with the new features/variables added. 
    Args: 
        dataframe: A pandas DataFrame object. 
    Return: 
        df: A pandas DataFrame object with the new features/variables added.
    '''
    for index, row in df.iterrows():  
        # Calculate and add new features to represent the bounding boxes
        transformed_boundingbox = transform_boundingbox(row['X-Min'], row['X-Max'], row['Y-Min'], row['Y-Max'], row['Z-Min'], row['Z-Max'],row['ox'],row['oy'],row['oz'],row['xx'],row['xy'],row['xz'],row['yx'],row['yy'],row['yz'],row['zx'],row['zy'],row['zz'])
        center_x, center_y, center_z = calculate_center_point(transformed_boundingbox)
        length, width, height = calculate_lwh(transformed_boundingbox=transformed_boundingbox)
        theta_x, theta_y, theta_z = calculate_orientation(transformed_boundingbox=transformed_boundingbox)

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
def outlier_detection(df_new_features: pd.DataFrame) -> pd.DataFrame:
    '''
    The function takes a pandas DataFrame as input and implements an outlier detection method to identify outliers in the "X-Max_transf" column. 
    It calculates the upper and lower limits, creates arrays of Boolean values indicating the outlier rows, and sets the bounding box features to zero if detected as an outlier. 
    The function returns the updated pandas DataFrame with the outliers removed/set to zero. 
    Args: 
        df_new_features: A pandas DataFrame object. 
    Return: 
        df_new_features: A pandas DataFrame object with the outlier bounding box features set to zero.
    '''
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
    df_new_features.loc[upper_array, config["general_params"]["bounding_box_features_original"]] = 0
    df_new_features.loc[lower_array, config["general_params"]["bounding_box_features_original"]] = 0

    return df_new_features
# %%
def preprocess_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    The function takes in a pandas DataFrame and performs various preprocessing steps on the data. 
    It fills the bounding box information, calculates features, drops outliers, selects new relevant features, removes data points from the front area of the car, cleans designations, removes mirrored car parts and removes duplicates. 
    It returns two DataFrames, one with the preprocessed data, and another with parts suitable for visualization purposes. 
    Args: 
        df: A pandas DataFrame object. 
    Return: 
        df_relevants: dataframe with the preprocessed data
        df_for_plot: dataframe with parts suitable for visualization purposes 
    '''
    logger.info(f"Start preprocessing the dataframe with {df.shape[0]} samples...")

    df.loc[df['X-Max'] == 10000, ['X-Min', 'X-Max', 'Y-Min', 'Y-Max', 'Z-Min', 'Z-Max', 'ox', 'oy', 'oz', 'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz', 'Wert']] = 0

    df_new_features = add_new_features(df)

    df_new_features = outlier_detection(df_new_features)

    # Using dictionary to convert specific columns
    df_new_features = df_new_features.astype(config["convert_dict"])

    # Select only car parts with bounding box information
    df_new_features = df_new_features[(df_new_features['X-Min'] != 0) & (df_new_features['X-Max'] != 0)]

    # Save the samples without/wrong bounding box information in a new df, as they will need to be added back later
    df_temp = df[(df["X-Min"] == 0.0) & (df["X-Max"] == 0.0)]

    # Delete all samples which have less volume than 500,000 mm^3
    df_relevants = df_new_features[(df_new_features['volume'] > 500000)].reset_index(drop=True)

    # Delete all samples where the parts are in the front area of the car
    x_min_transf, x_max_transf = df_relevants["X-Min_transf"].min(), df_relevants["X-Max_transf"].max()
    car_length = x_max_transf - x_min_transf
    cut_point_x = x_min_transf + car_length*config["general_params"]["cut_percent_of_front"]
    df_relevants = df_relevants[df_relevants["X-Min_transf"] > cut_point_x]

    # Concatenate the two dataframes
    df_relevants = pd.concat([df_relevants, df_temp], ignore_index=True).reset_index(drop=True)

    # Clean the designations and store the result in the column "Benennung (bereinigt)"
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
        # Get list of unique values in column "Einheitsname"
        unique_einheitsnamen = np.unique(y)
        weight_factor = {}
        for name in unique_einheitsnamen:
            weight_factor[name] = round(np.count_nonzero(y != name) / np.count_nonzero(y == name))
            if weight_factor[name] == 0:
                weight_factor[name] = 1
    else:
        if df[df["Relevant fuer Messung"]=="Ja"].shape[0] == 0:
            weight_factor = 0
            logger.error("The dataset does not contain any ""Ja"" labeled samples")
        else:
            weight_factor = round(df[df["Relevant fuer Messung"]=="Nein"].shape[0] / df[df["Relevant fuer Messung"]=="Ja"].shape[0])

    return weight_factor

# %%
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
    
    X = nchar_text_to_vec(data=df, model_folder_path=model_folder_path) # Using ngram vectorizer
    #X = doc2vec_text_to_vec(df, model_folder_path)
    #X = bert_text_to_vec(df, model_folder_path)

    # Combine text features with other features
    features = config["general_params"]["features_for_model"]
    bbox_features_dict = {"features_for_model": features}
    with open(model_folder_path + 'boundingbox_features.pkl', 'wb') as fp:
        pickle.dump(bbox_features_dict, fp)

    if config["general_params"]["use_only_text"] == False:
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

    weight_factor = get_weight_factor(y=y, df=df, binary_model=binary_model)     

    indices = np.arange(X.shape[0])
    X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(X, y, indices, test_size=config["train_settings"]["train_val_split"], stratify=y, random_state=config["general_params"]["seed"])
    X_val, X_test, y_val, y_test, indices_val, indices_test = train_test_split(X_val, y_val, indices_val, test_size=config["train_settings"]["val_test_split"], stratify=y_val, random_state=config["general_params"]["seed"])

    df_train = df.iloc[indices_train]
    df_val = df.iloc[indices_val]
    df_test = df.iloc[indices_test]

    logger.success("Train, validation and test sets are generated!")

    return X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test, weight_factor

def main():
    
    test_preprocessing = True
    data_path = 'C:/Users/q617269/Desktop/Masterarbeit_Tobias/repos/master-thesis/data/raw_for_labeling/prismaexport-20230731-163448.xls'
    #data_path = "C:/Users/q617269/Desktop/Masterarbeit_Tobias/repos/master-thesis/data/raw_for_labeling/prismaexport-20230731-171755.xls"

    if test_preprocessing:
        df = pd.read_excel(data_path, header=None, skiprows=1)
        df.columns = df.iloc[0]
        df = df.iloc[1:] 
        df, ncar = prepare_and_add_labels(df)
        df.to_excel("test_prepare_and_add_labels_func.xlsx")

        df_relevants, df_for_plot = preprocess_dataset(df)
        df_relevants.to_excel("test_preprocess_dataset_func.xlsx")

# %%
if __name__ == "__main__":
    
    main()