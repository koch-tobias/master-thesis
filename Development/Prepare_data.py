# %%
import pandas as pd
import os
import re
from loguru import logger
from pathlib import Path
from datetime import datetime
import shutil
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle

# %% [markdown]
# ### Functions

# %%
def load_csv_into_df(folder_name: Path, original_prisma_data: bool, move_to_archive: bool) -> list:
    '''
    This function searches for all .xls files in a given directory, loads each file into a Pandas dataframe and changes the header line.
    If move_to_archive is set True, then all processed files will be moved to the archive.
    return: List with all created dataframes
    '''
    # Check if the folder exists
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

                    if move_to_archive == True:
                        # Move file to archive
                        shutil.move(os.path.join(folder_name, file), os.path.join(folder_name, "original_data_archive", file))

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
def df_info_to_excel(df: pd.DataFrame):
    '''
    This function saves feature informations in an excel file
    '''
    pd.DataFrame({"name": df.columns, "non-nulls": len(df)-df.isnull().sum().values, "nulls": df.isnull().sum().values, "type": df.dtypes.values}).to_excel("data_infos.xlsx")

# %%
def prepare_and_add_labels(dataframes: list, save_as_excel: bool):

    logger.info("Start preprocessing the data...")
    dataframes_with_labels = []
    ncars = []

    for i in range(len(dataframes)):
        # Store the ncar abbreviation for file paths
        ncar = dataframes[i]['Benennung (dt)'][1][:3]
        ncars.append(ncar)

        # Temporary store the modul for the interior mirror
        level_interor_mirror = dataframes[i][dataframes[i]['Benennung (dt)'].str.startswith(f'{ncar} CE05')]["Ebene"].values[0]
        startindex_interor_mirror = dataframes[i][dataframes[i]['Benennung (dt)'].str.startswith(f'{ncar} CE05')].index[-1]+1
        endindex_interor_mirror = dataframes[i].loc[(dataframes[i]["Ebene"] == level_interor_mirror) & (dataframes[i].index > startindex_interor_mirror)].index[0]-1
        temp_interor_mirror = dataframes[i].loc[startindex_interor_mirror:endindex_interor_mirror]

        # Temporary store the modul for the roof antenna
        level_roof_antenna = dataframes[i][dataframes[i]['Benennung (dt)'].str.startswith(f'{ncar} CD07')]["Ebene"].values[0]
        startindex_roof_antenna = dataframes[i][dataframes[i]['Benennung (dt)'].str.startswith(f'{ncar} CD07')].index[-1]+1
        endindex_roof_antenna = dataframes[i].loc[(dataframes[i]["Ebene"] == level_roof_antenna) & (dataframes[i].index > startindex_roof_antenna)].index[0]-1
        temp_roof_antenna = dataframes[i].loc[startindex_roof_antenna:endindex_roof_antenna]

        # Keep only car parts of module group EP
        index_EF_module = dataframes[i][dataframes[i]['Benennung (dt)'].str.startswith(f'EF {ncar}')].index[-1]
        dataframes[i] = dataframes[i].loc[:index_EF_module-1]

        # Add interor mirror 
        dataframes[i] = pd.concat([dataframes[i], temp_interor_mirror]).reset_index(drop=True)
    
        # Add roof antenna 
        dataframes[i] = pd.concat([dataframes[i], temp_roof_antenna]).reset_index(drop=True)

        # Keep only the relevant samples with Dok-Format=5P. This samples are on the last level of the car structure
        dataframes[i] = dataframes[i][dataframes[i]["Dok-Format"]=='5P'].reset_index(drop=True)

        # Delete the NCAR abbreviation because of data security reasons
        dataframes[i]["Benennung (dt)"] = dataframes[i]["Benennung (dt)"].apply(lambda x: x.replace(ncar, ""))

        # Keep only features which are identified as relevant for the preprocessing, the predictions or for the users' next steps
        dataframes[i] = dataframes[i][['Sachnummer','Benennung (dt)', 'X-Min','X-Max','Y-Min','Y-Max','Z-Min','Z-Max', 'Wert','Einheit','Gewichtsart','Kurzname','L-Kz.', 'L/R-Kz.', 'Modul (Nr)', 'ox','oy', 'oz', 'xx','xy','xz', 'yx','yy','yz','zx','zy','zz']]

        # using dictionary to convert specific columns
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
        
        dataframes[i] = dataframes[i].astype(convert_dict)

        # Add columns for the label "Relevant für Messung" and "Allgemeine Bezeichnung"
        data_labeled = dataframes[i]
        data_labeled.insert(len(data_labeled.columns), 'Relevant fuer Messung', 'Nein')
        data_labeled.insert(len(data_labeled.columns), 'Einheitsname', 'Dummy')
        dataframes_with_labels.append(data_labeled)

        if save_as_excel==True:
            # Date
            dateTimeObj = datetime.now()
            timestamp = dateTimeObj.strftime("%d%m%Y_%H%M")
            
            # Store preprocessed dataframes
            dataframes_with_labels[i].to_excel(f"../data/preprocessed_data/{ncar}_preprocessed_{timestamp}.xlsx")

    if save_as_excel == True:
        logger.success(f"The features are reduced and formated to the correct data type. The new dataset is stored as {ncar}_preprocessed_{timestamp}.xlsx!")
    else:
        logger.success(f"The features are reduced and formated to the correct data type!")
    
    return dataframes_with_labels, ncars


# %%
def prepare_text(designation: str) -> str:
    # transform to lower case
    text = str(designation).upper()

    # Removing punctations
    text = re.sub(r"[^\w\s]", "", text)

    # Removing numbers
    text = ''.join([i for i in text if not i.isdigit()])

    # tokenize text
    text = text.split(" ")

    # Remove predefined words
    predefined_words = ['ZB', 'AF', 'LI', 'RE', 'MD', 'LL', 'TAB', 'TB']
    if len(predefined_words) > 0:
        text = [word for word in text if word not in predefined_words]

    # Remove words with only one letter
    text = [word for word in text if len(word) > 1]

    # remove empty tokens
    text = [t for t in text if len(t) > 0]

    # join all
    prepared_designation = " ".join(text)

    return prepared_designation

# %%
def vectorize_data(data: pd.DataFrame, df_val, timestamp) -> tuple:
    #token = WhitespaceTokenizer()
    #vectorizer = TfidfVectorizer(analyzer="word", tokenizer=token.tokenize)

    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 8))

    X_text = vectorizer.fit_transform(data['Benennung (bereinigt)']).toarray()
    X_test = vectorizer.transform(df_val['Benennung (bereinigt)']).toarray()

    # Store the vocabulary
    vocabulary = vectorizer.get_feature_names_out()

    # Save the vectorizer and vocabulary to files
    os.makedirs(f'../models/lgbm_{timestamp}')
    with open(f'../models/lgbm_{timestamp}/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(f'../models/lgbm_{timestamp}/vocabulary.pkl', 'wb') as f:
        pickle.dump(vocabulary, f)

    return X_text, X_test

# %%
def clean_text(df):
    df["Benennung (bereinigt)"] = df.apply(lambda x: prepare_text(x["Benennung (dt)"]), axis=1)

    return df

# %%
def train_test_val(df, df_test,only_text: bool, test_size:float, timestamp):
    
    X, X_test = vectorize_data(df, df_test, timestamp)

    # Combine text features with other features
    features = ['center_x', 'center_y', 'center_z','length','width','height','theta_x','theta_y','theta_z']
    if only_text == False:
        X = np.concatenate((X, df[features].values), axis=1)
        X_test = np.concatenate((X_test, df_test[features].values), axis=1)

    y = df['Relevant fuer Messung']
    y = y.map({'Ja': 1, 'Nein': 0})

    y_test = df_test['Relevant fuer Messung']
    y_test = y_test.map({'Ja': 1, 'Nein': 0})

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test, features

# %%
def train_test_val_kfold(df, df_test, timestamp):
    #df["Benennung (dt)"] = df.apply(lambda x: prepare_text(x["Benennung (dt)"]), axis=1)
    #df_test["Benennung (dt)"] = df_test.apply(lambda x: prepare_text(x["Benennung (dt)"]), axis=1)

    X, X_test = vectorize_data(df, df_test, timestamp)

    # Combine text features with other features
    features = ['center_x', 'center_y', 'center_z','length','width','height','theta_x','theta_y','theta_z']
    #X = np.concatenate((X, df[features].values), axis=1)

    y = df['Relevant fuer Messung']
    y = y.map({'Ja': 1, 'Nein': 0})

    y_test = df_test['Relevant fuer Messung']
    y_test = y_test.map({'Ja': 1, 'Nein': 0})

    return X, y, X_test, y_test, features

# %% [markdown]
# ### Main

# %%
def main():
    # Define the path to the folder containing the data (xls files)
    data_folder = Path("../data/labeled_data")
    dataframes = load_csv_into_df(data_folder, original_prisma_data=False, move_to_archive=False)
    df = combine_dataframes(dataframes)


# %%
if __name__ == "__main__":
    
    main()


