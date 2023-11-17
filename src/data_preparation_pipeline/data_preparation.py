import pandas as pd

from loguru import logger
import yaml
from yaml.loader import SafeLoader

with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

class Preperator:
    """
    This class creates a instance to select the pre-defined modules and keep only components of the labeled data (structural part lists).
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
        This function searches the index of the last record in a pre-defined module.
        Args:
            lst: list which contains all records which are on the same level
            value: index of the record where the desired module starts
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
        This function gets a dataframe- (structural part list) as input and returns a new dataframe with only pre-defined modules.
        Args:
            dataframe: The initial dataframe containing the structural part list with all car parts the hierachical structures and the metadata.
        Return:
            dataframe_new: A new dataframe containing only component of pre-defined modules.
        '''
        # Initialize an empty dataframe with the same columns as the given one
        dataframe_new = pd.DataFrame(columns=dataframe.columns)

        # Iterate over each module which should be kept in the data
        for module in config["keep_modules"]:
            endindex = -1
            try: 
                # Iterate over each record to find the start and endindex of the pre-defined modules. Then, store  the record in a new dataframe
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

        # Keep only components and remove all records which represent the hierarchical structure
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
        features = config["relevant_features"] + [config["labels"]["binary_column"]] + [config["labels"]["multiclass_column"]] + ['Derivat']

        # Keep only features which are identified as relevant for the preprocessing, the predictions or for the users' next steps
        dataframe_new = dataframe[features]
        
        # Convert the data types of the numerical features 
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

    @staticmethod
    def data_preparation(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        '''
        The function takes a pandas DataFrame as input and prepares the data. 
        It drops all empty columns, checks if all relevant features are available, stores the NCAR abbreviation for file paths, retains only the relevant modules and only keep relevant features. 
        Finally, it returns a tuple with the preprocessed DataFrame object and the NCAR abbreviation. 
        Args: 
            dataframe: A pandas DataFrame object. 
        Return: 
            dataframe: preprocessed pandas DataFrame object 
            ncar: string (NCAR abbreviation) which is used for file paths.
        '''

        # Check if all necassary columns are available in the dataset
        missing_columns = Preperator.check_if_columns_available(dataframe=dataframe, relevant_features=config["relevant_features"])
        if len(missing_columns) > 0:
            logger.exit(f"Please check your dataset. The following attributes are missing: {missing_columns}")
        
        # Get the derivat of the selected car
        ncar = dataframe.iloc[1]['Derivat']

        # Select pre-defined modules 
        dataframe_new = Preperator.car_part_selection(dataframe)

        # Delete the NCAR abbreviation due to data privacy
        dataframe_new[config["dataset_params"]["car_part_designation"]] = dataframe_new[config["dataset_params"]["car_part_designation"]].apply(lambda x: x.replace(ncar, ""))
        
        # Select the pre-defined features, which are relevant for further preprocessing
        dataframe_new = Preperator.feature_selection(dataframe_new)
        
        return dataframe_new, ncar