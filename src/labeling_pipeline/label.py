import pandas as pd

import os
import shutil
from loguru import logger

import yaml
from yaml.loader import SafeLoader
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

import sys
sys.path.append(config['paths']['project_path'])
from src.deployment_pipeline.prediction import predict_on_new_data

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
                df.at[i, column] = modified_value
    
    return df


def load_data_into_df() -> tuple[list, str]:
    ''' 
    This function loads data from the specified folder path. It reads data from all files in the folder, converts them to pandas dataframes and stores the dataframes in a list. 
    The list of dataframes is returned as output. 
    Args:
        None
    Return:
        dataframes: a list containing pandas dataframes of all the files read from the specified folder path
    '''

    # Check if the folder exists
    folder_name = "data/raw_for_labeling"
        
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
                    df = pd.read_excel(os.path.join(folder_name, file), header=None, skiprows=1)
                    df.columns = df.iloc[0]
                    df = df.iloc[1:]
                    # Drop all empty columns
                    dataframe = df.dropna(how= "all", axis=1, inplace=False)
                    # Store the ncar abbreviation for file paths
                    ncar = dataframe['Code'].iloc[0]

                    df = prepare_columns(df)
                    # Add the created dataframe to the list of dataframes
                    dataframes.append(df)
                    ncars.append(ncar)

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

        return dataframes 
    
def label_data() -> None:
    ''' 
    Label the data by predicting relevant car parts and unique names on the new dataset. It saves the labeled data as a CSV file in the specified folder. 
    If unique names are not found, it asks to add them manually.
    Args: None
    Return: None 
    '''
    dataframes = load_data_into_df()
    for df in dataframes:
        df_with_label_columns, df_relevant_parts, einheitsname_not_found, ncar = predict_on_new_data(df=df)

        for index, row in df_relevant_parts.iterrows():
            sachnummer = row['Sachnummer']
            einheitsname = row['Einheitsname']
            
            if sachnummer in df['Sachnummer'].values:
                df_with_label_columns.loc[df_with_label_columns['Sachnummer'] == sachnummer, 'Relevant fuer Messung'] = "Ja"
                df_with_label_columns.loc[df_with_label_columns['Sachnummer'] == sachnummer, 'Einheitsname'] = einheitsname

        features = config["general_params"]["relevant_features"] + ['Relevant fuer Messung','Einheitsname']
        df_with_label_columns = df_with_label_columns[features]
        df_with_label_columns.to_csv(f"data/pre_labeled/{ncar}_labeled.csv")

        logger.info(f"The following car parts are not found in your dataset: {einheitsname_not_found} If essential, please add this car parts manually!")
        logger.success(f"The prediction is done and the result is stored here: data/pre_labeled_data/{ncar}_labeled.csv!")

# %%
def main():
    label_data()

# %%
if __name__ == "__main__":
    
    main()