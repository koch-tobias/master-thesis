# %%
import pandas as pd
import numpy as np

import re
import pickle
from loguru import logger

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

import yaml
from yaml.loader import SafeLoader
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

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
    if not df_new_features.empty:
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
def prepare_text(designation: str) -> str:
    ''' 
    This function takes in a string, performs a series of text preprocessing tasks, and returns the resulting cleaned string. 
    The tasks it performs include converting all characters to uppercase, removing all punctuation marks, removing all numeric digits, removing predefined words, removing all words with only one letter, and removing all empty tokens. 
    Args:
        designation: A string that needs to be prepared. 
    Return:
        designation: The function returns a string which is the cleaned version of the original input string. 
    '''
    # transform to lower case
    text = str(designation).upper()

    # Removing punctations
    text = re.sub(r"[^\w\s]", " ", text)

    # Removing numbers
    text = ''.join([i for i in text if not i.isdigit()])

    # tokenize text
    text = text.split(" ")

    # Remove predefined words
    predefined_words = ["ZB", "AF", "LI", "RE", "MD", "LL", "TAB", "TB"]
    text = [word for word in text if word not in predefined_words]

    # Remove words with only one letter
    text = [word for word in text if len(word) > 1]

    # remove empty tokens
    text = [word for word in text if len(word) > 0]

    # join all
    prepared_designation = " ".join(text)

    return prepared_designation

# %%
def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    ''' 
    Description: Cleans text data in the DataFrame by applying the 'prepare_text' function on the 'Benennung (dt)' column, and adds the cleaned text data as a new column, 'Benennung (bereinigt)'.
    Args:
        df: DataFrame containing 'Benennung (dt)' column
    Return:
        df: DataFrame with an additional cleaned text column, 'Benennung (bereinigt)' 
    '''
    df["Benennung (bereinigt)"] = df.apply(lambda x: prepare_text(x[config['general_params']['car_part_designation']]), axis=1) 

    return df

# %%
def nchar_text_to_vec(data: pd.DataFrame, model_folder_path: str) -> tuple:
    '''
    This function converts text data into vector representation using the n-gram approach.
    Args:
        data (pd.DataFrame): The input DataFrame containing the text data.
        model_folder_path (str): The path to the folder where the model files will be saved.
    Returns:
        tuple: A tuple containing the vectorized text data.
    '''

    # Initialize the CountVectorizer with the desired settings
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 8), max_features=8000)

    # Convert the text data into a vector representation
    X_text = vectorizer.fit_transform(data['Benennung (bereinigt)']).toarray()

    # Store the vocabulary
    vocabulary = vectorizer.get_feature_names_out()

    # Save the vectorizer and vocabulary if a model folder path is provided
    if model_folder_path != "":
        with open(model_folder_path + 'vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(model_folder_path + 'vocabulary.pkl', 'wb') as f:
            pickle.dump(vocabulary, f)

    # Return the vectorized text data
    return X_text


# %%
def get_vocabulary(column) -> list:
    '''
    This function extracts the vocabulary from a given column of text data.
    Args:
        column: The input column containing the text data.
    Returns:
        list: A list of unique words in the text data.
    '''

    # Concatenate all the text data into a single string
    text = ' '.join(column.astype(str))

    # Split the text into individual words and convert them to uppercase
    words = text.upper().split()

    # Count the occurrences of each word and sort them in descending order
    word_counts = pd.Series(words).value_counts()

    # Extract the unique words as the vocabulary
    vocabulary = word_counts.index.tolist()

    # Return the vocabulary
    return vocabulary


# %%
def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    df_new_features = outlier_detection(df)

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
        binary_label_column = config['labels']['binary_column']
        if df[df[binary_label_column]==config['labels']['binary_label_1']].shape[0] == 0:
            weight_factor = 0
            logger.error("The dataset does not contain any ""Ja"" labeled samples")
        else:
            weight_factor = round(df[df[binary_label_column]==config['labels']['binary_label_0']].shape[0] / df[df[binary_label_column]==config['labels']['binary_label_1']].shape[0])

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
    
    X = nchar_text_to_vec(data=df, model_folder_path=model_folder_path)

    # Combine text features with other features
    features = config["general_params"]["features_for_model"]
    bbox_features_dict = {"features_for_model": features}
    with open(model_folder_path + 'boundingbox_features.pkl', 'wb') as fp:
        pickle.dump(bbox_features_dict, fp)

    if config["general_params"]["use_only_text"] == False:
        X = np.concatenate((X, df[features].values), axis=1)

    if binary_model:
        y = df[config['labels']['binary_column']]
        y = y.map({config['labels']['binary_label_1']: 1, config['labels']['binary_label_0']: 0})
    else:
        y = df[config['labels']['multiclass_column']] 
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
    print("Empty main!")

# %%
if __name__ == "__main__":
    
    main()