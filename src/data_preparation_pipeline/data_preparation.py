import pandas as pd
from loguru import logger
import yaml
from yaml.loader import SafeLoader

with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

class Preperator:
    """
    This class creates a instance to prepare raw data.
    """

    # PyTest exist
    @staticmethod
    def check_if_columns_available(dataframe: pd.DataFrame, relevant_features: list) -> list:
        '''
        The function takes a pandas DataFrame and a list of relevant features as input. 
        It checks if all relevant features are present in the input DataFrame and returns a list of missing features. 
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

    @staticmethod
    def get_endindex(lst: list, value):
        """
        This function takes a list and a value as input, and returns the next value
        after the given one.
        Args:
            lst: list 
            value: value which will be searched in the list
        Return: 
            Next value in list or None (if not found or last element in list)
        """
        # Get the index of the given value in the list
        try:
            index = lst.index(value)
        except ValueError:
            return None
        
        # If the given value is the last element in the list, return None
        if index == len(lst) - 1:
            return None
        
        # Otherwise, return the next value in the list
        return lst[index + 1]
        
    # PyTest exist
    @staticmethod
    def car_part_selection(dataframe: pd.DataFrame) -> pd.DataFrame:
        '''
        This function selects the car parts from a dataframe of car structure and returns a new dataframe with only the selected modules.
        Args:
            dataframe: The initial dataframe containing the car structure with all car parts and the metadata.
        Return:
            dataframe_new: A new dataframe containing only the car parts on the last level of the car structure with "Dok-Format" equals to "5P" and the selected modules.
        '''
        # Initialize an empty dataframe with the same columns as the given one
        dataframe_new = pd.DataFrame(columns=dataframe.columns)

        # Iterate over each module which should be kept
        for module in config["keep_modules"]:
            endindex = -1
            try: 
                # Iterate over each sample where the module number equals the module which should be kept.
                # It stores all car parts which are in the module to the new dataframe
                for i in range(dataframe[dataframe["Modul (Nr)"] == module].shape[0]):
                    level = dataframe[dataframe["Modul (Nr)"] == module]["Ebene"].values[i]
                    startindex = dataframe[dataframe["Modul (Nr)"] == module].index[i]
                    if startindex > endindex:
                        try:
                            modules_on_same_level = dataframe[dataframe["Ebene"] == level].index.tolist()
                            endindex = Preperator.get_endindex(modules_on_same_level, startindex)
                        except: 
                            endindex = dataframe.shape[0] + 1
                        temp = dataframe.loc[startindex:endindex]
                        dataframe_new = pd.concat([dataframe_new, temp], ignore_index=True).reset_index(drop=True)

            except:
                logger.info(f"Module {module} not found in the structure tree!")

        # Keep only the relevant samples with Dok-Format=5P. These are on the last level of the car structure and contains only car parts
        dataframe_new = dataframe_new[dataframe_new["Dok-Format"]=='5P'].reset_index(drop=True)

        return dataframe_new

    # PyTest exist
    @staticmethod
    def feature_selection(dataframe: pd.DataFrame) -> pd.DataFrame:
        '''
        This function selects the relevant features from a given dataframe. It then converts the numerical features to float values.
        Args:
            dataframe: The initial dataframe containing possibly irrelevant features.
        Return:
            dataframe_new: A new dataframe with only relevant features and converted datatypes.
        '''
        # Keep only features which are identified as relevant for the preprocessing, the predictions or for the users' next steps
        dataframe_new = dataframe[config["relevant_features"]]
        
        # Dict which is used to transform the data types of the bounding box features 
        convert_dict = {
                        "X-Min": float,
                        "X-Max": float,
                        "Y-Min": float,
                        "Y-Max": float,
                        "Z-Min": float,
                        "Z-Max": float,
                        "Wert": float,
                        "ox": float,
                        "oy": float,
                        "oz": float,
                        "xx": float,
                        "xy": float,
                        "xz": float,
                        "yx": float,
                        "yy": float,
                        "yz": float,
                        "zx": float,
                        "zy": float,
                        "zz": float
                    }           

        dataframe_new = dataframe_new.astype(convert_dict)

        return dataframe_new

    # PyTest exist
    @staticmethod
    def add_labels(dataframe: pd.DataFrame) -> pd.DataFrame:
        '''
        This function adds the label columns to the dataframe.
        Args:
            dataframe: The initial dataframe containing possibly irrelevant features.
        Return:
            dataframe_new: A new dataframe with the added labels.
        '''
        # Add and initialize the label columns "Relevant für Messung" and "Einheitsname"
        dataframe.insert(len(dataframe.columns), config['labels']['binary_column'], config['labels']['binary_label_0']) 
        dataframe.insert(len(dataframe.columns), config['labels']['multiclass_column'], 'Dummy') 

        return dataframe

    @staticmethod
    def data_preparation(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, str]:
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

        # Check if all necassary columns are available in the dataset
        missing_columns = Preperator.check_if_columns_available(dataframe=dataframe, relevant_features=config["relevant_features"])
        if len(missing_columns) > 0:
            logger.exit(f"Please check your dataset. The following attributes are missing: {missing_columns}")

        # Get the derivat of the selected car
        ncar = dataframe.iloc[1]['Code']

        # Select specified modules 
        dataframe_new = Preperator.car_part_selection(dataframe)

        # Delete the NCAR abbreviation due to data security
        dataframe_new[config["dataset_params"]["car_part_designation"]] = dataframe_new[config["dataset_params"]["car_part_designation"]].apply(lambda x: x.replace(ncar, ""))
        
        # Select relevant features
        dataframe_new = Preperator.feature_selection(dataframe_new)

        # Add label columns
        dataframe_new = Preperator.add_labels(dataframe_new)

        # Reset and drop index
        dataframe_new = dataframe_new.reset_index(drop=True)

        logger.success(f"The data is successfully prepared! The features are reduced and formated to the correct data type, subfolders are deleted, and only relevant modules are kept!")
        
        return dataframe_new, ncar