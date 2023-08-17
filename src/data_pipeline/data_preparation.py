import pandas as pd

from loguru import logger

import yaml
from yaml.loader import SafeLoader
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
    
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