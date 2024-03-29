import pandas as pd
import numpy as np

from loguru import logger
from pathlib import Path
import pickle
import os
import yaml
from yaml.loader import SafeLoader
import sys
sys.path.append(os.getcwd())

from src.data_preparation_pipeline.data_preparation import Preperator
from src.data_preprocessing_pipeline.feature_engineering import Feature_Engineering
from src.data_preprocessing_pipeline.data_cleaning import DataCleaner
from src.utils import read_file, add_labels


with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

class Identifier():

    @staticmethod
    def search_in_logging(text:str, model_folder_path: Path) -> str or None:
        ''' 
        Returns the value after a searched text by reading the 'logging.txt' file of the given model path.
        Args:
            text: string which should be searched in the logging file
            model_folder_path: path of folder containing the model
        Return:
            value after the searched string
            None if the string is not found in the 'logging.txt' file 
        '''
        logging_path = os.path.join(model_folder_path, "logging.txt")
        with open(logging_path, 'r') as file:
            for line in file:
                if text in line:
                    return line.split(":")[1].strip()
        return None

    @staticmethod
    def get_model(folder_path: Path):
        '''
        Search in the model folder path for the trained models and load the vectorizer, vocabulary, and boundingbox_features used for training.
        Args:
            folder_path = path to the folder where the models are stored
        Return:
            model = final trained model
            vectorizer = vectorizer used to train the final model
            vocabulary = vocabulary used to train the final model
            bbox_features = bounding box features used to train the final model
        '''
        final_model_path = os.path.join(folder_path, "final_model.pkl")
        pretrained_model_path = os.path.join(folder_path, "model.pkl")

        if os.path.exists(final_model_path):
            model_path =  final_model_path
        else:
            model_path =  pretrained_model_path

        with open(model_path, "rb") as fid:
            model = pickle.load(fid)

        # Get the dataset path from the logging file
        dataset_path = Identifier.search_in_logging(text="Dataset:", model_folder_path=folder_path)

        # Load the vectorizer from the file
        vectorizer_path = os.path.join(dataset_path, "vectorizer.pkl")
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

        # Get the vocabulary of the training data
        vocab_path = os.path.join(dataset_path, "vocabulary.pkl")
        with open(vocab_path, 'rb') as f:
            vocabulary = pickle.load(f) 

        # Get the used boundingbox features
        bbox_features_path = os.path.join(dataset_path, "boundingbox_features.pkl")
        with open(bbox_features_path, 'rb') as f:
            bbox_features = pickle.load(f)  

        return model, vectorizer, vocabulary, bbox_features

    @staticmethod
    def get_X(vocab, vectorizer, bbox_features: list, df_preprocessed: pd.DataFrame, model_folder_path: Path) -> np.array:
        '''
        Prepare the text data and combine with additional features to generate the dataset to train the model 
        Args:
            vocab = vocabulary of the dataset
            vectorizer = vectorizer trained on the dataset
            bbox_features = list of bounding box features which should be used additionally to the designation 
            df_preprocessed = dataframe with the preprocessed data
            model_folder_path = Path to the folder of the final model
        Return:
            X = dataset prepared to train the model
        '''
        # Convert the vocabulary list to a dictionary
        vocabulary_dict = {word: index for index, word in enumerate(vocab)}

        # Set the vocabulary of the vectorizer to the loaded vocabulary
        vectorizer.vocabulary_ = vocabulary_dict
        X = vectorizer.transform(df_preprocessed['Benennung (bereinigt)']).toarray()

        use_only_text = Identifier.search_in_logging(text="use_only_text:", model_folder_path=model_folder_path)

        # Combine text features with other features
        if use_only_text == "False":      
            X = np.concatenate((X, df_preprocessed[bbox_features].values), axis=1)
        
        return X

    @staticmethod
    def get_best_iteration(model, method: str) -> int:
        '''
        Get the best iteration of the trained model
        Args:
            model = trained model
            method = string which method is used for training ("lgbm", "xgboost", "catboost")
        Return:
            best_iteration
        '''
        if method == "lgbm":
            best_iteration = model._best_iteration - 1
        elif method == "xgboost":
            best_iteration = model.get_booster().best_ntree_limit - 1
        elif method == "catboost":
            best_iteration = model.get_best_iteration()

        return best_iteration

    @staticmethod
    def get_probabilities(model, X_test: np.array, best_iteration: int, method: str) -> np.array:
        '''
        Get the probibilities for the model prediction
        Args:
            model = trained model
            X_test = test set
            best_iteration = best iteration of the trained model
            method = string which method is used for training ("lgbm", "xgboost", "catboost")
        Return:
            probs = probabilities
        '''
        if method == "lgbm":
            probs = model.predict_proba(X_test, num_iteration=best_iteration)
        elif method == "xgboost":
            probs = model.predict_proba(X_test, ntree_limit=best_iteration)
        elif method == "catboost":
            probs = model.predict_proba(X_test)

        return probs

    # %%
    @staticmethod
    def model_predict(model, X_test: np.array, method: str, binary_model: bool) -> tuple[np.array, np.array, int]:
        ''' 
        Returns predicted output, probabilities, and best iteration number of a given machine learning model on test data using the chosen method. If binary_model flag is True, it predicts binary output; otherwise, it predicts the class with the highest probability.

        Args:
            model: machine learning model
            X_test: test data
            method: str, the chosen method for identifying best iteration
            binary_model: bool, indicates whether binary classification or multiclass classification is expected
        Return:
            y_pred: predicted output
            probs: probability values
            best_iteration: the best iteration number based on the chosen method 
        '''
        best_iteration = Identifier.get_best_iteration(model=model, method=method)
        probs = Identifier.get_probabilities(model=model, X_test=X_test, best_iteration=best_iteration, method=method)

        if binary_model:
            # Predict that the car part is relevant if the probability is above the threshold
            y_pred = (probs[:,1] >= config["prediction_settings"]["prediction_threshold"])
            y_pred =  np.where(y_pred, 1, 0)
        else:
            y_pred = probs.argmax(axis=1)   
        
        return y_pred, probs, best_iteration

    @staticmethod
    def classification_on_new_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list, str]:
        ''' 
        Predicts relevant car parts and unique names on a new dataset by using a binary model and a multiclass model. 
        It also validates the identified car parts for their position and compatibility with the training set. 
        Then, the function prepares the output as DataFrames and a list, and returns a tuple containing four values: preprocessed DataFrame, DataFrame of relevant car parts with unique names, a list of unique names that were not found, and the 'ncar' value which is the car derivat.

        Args:
            df: new dataset
        Return:
            df_preprocessed: preprocessed DataFrame for the input data
            df_relevant_parts: DataFrame containing relevant car parts with their unique names and additional information about them
            einheitsname_not_found: list of unique names not found
            ncar: number of recognized VINs 
        '''
        logger.info("Prepare dataset...")
        df, ncar = Preperator.data_preparation(dataframe=df)
        logger.info("Dataset successfully prepared!")

        logger.info("Add label columns...")
        dataframe_new = add_labels(dataframe_new)
        dataframe_new = dataframe_new.reset_index(drop=True)
        logger.info("Labels added successfully!")

        logger.info("Preprocess data...")
        
        df_new_features = Feature_Engineering.add_new_features(df)
        
        df_preprocessed, df_for_plot = DataCleaner.clean_dataset(df_new_features)
        logger.info("Dataset successfully preprocessed!")

        logger.info("Load pretrained models...")
        model_folder_path_binary = Path("final_models/Binary_model")
        model_folder_path_multiclass = Path("final_models/Multiclass_model")
        model_binary, vectorizer_binary, vocabulary_binary, boundingbox_features_binary = Identifier.get_model(folder_path=model_folder_path_binary)
        model_multiclass, vectorizer_multiclass, vocabulary_multiclass, boundingbox_features_multiclass = Identifier.get_model(folder_path=model_folder_path_multiclass)       
        logger.success("Pretrained models are loaded!")

        # Transform the data in the same way the model is trained
        X_binary = Identifier.get_X(vocab=vocabulary_binary, vectorizer=vectorizer_binary, bbox_features=boundingbox_features_binary["features_for_model"], df_preprocessed=df_preprocessed, model_folder_path=model_folder_path_binary)
        X_multiclass = Identifier.get_X(vocab=vocabulary_multiclass, vectorizer=vectorizer_multiclass, bbox_features=boundingbox_features_multiclass["features_for_model"], df_preprocessed=df_preprocessed, model_folder_path=model_folder_path_multiclass)

        logger.info("Identify relevant car parts and add their uniform names...")
        binary_method = Identifier.search_in_logging(text="Method:", model_folder_path=model_folder_path_binary)
        multiclass_method = Identifier.search_in_logging(text="Method:", model_folder_path=model_folder_path_multiclass)
        y_pred_binary, probs_binary, _  = Identifier.model_predict(model=model_binary, X_test=X_binary, method=binary_method, binary_model=True)
        y_pred_multiclass, probs_multiclass, _ = Identifier.model_predict(model=model_multiclass, X_test=X_multiclass, method=multiclass_method, binary_model=False)

        # Load the LabelEncoder
        label_encoder_path = os.path.join(model_folder_path_multiclass, "label_encoder.pkl")
        with open(label_encoder_path, 'rb') as f:
            le = pickle.load(f) 

        y_pred_multiclass_names = le.inverse_transform(y_pred_multiclass) 

        df_relevant_parts = df_preprocessed.reset_index(drop=True)

        for index, row in df_relevant_parts.iterrows():
            if y_pred_binary[index] == 1: 
                df_relevant_parts.loc[index,'Relevant fuer Messung'] = 'Ja'
            else:
                df_relevant_parts.loc[index,'Relevant fuer Messung'] = 'Nein'

            df_relevant_parts.loc[index,'Einheitsname'] = y_pred_multiclass_names[index]
            df_relevant_parts.loc[index,'Wahrscheinlichkeit Relevanz'] = probs_binary[:, 1][index]
            df_relevant_parts.loc[index,'Wahrscheinlichkeit Einheitsname'] = probs_multiclass[index, y_pred_multiclass[index]]

        df_relevant_parts = df_relevant_parts[df_relevant_parts['Relevant fuer Messung'] == 'Ja']
        logger.success("Relevant car parts are identified!")
        
        logger.info("Prepare output...")
        # Load list of the uniform names (classes)
        with open(os.path.join(model_folder_path_binary, "list_of_uniform_names.pkl"), 'rb') as names:
            uniform_names = pickle.load(names)

        # Check which uniform names are not identified
        einheitsname_not_found = []
        for name in uniform_names:        
            if name not in df_relevant_parts['Einheitsname'].unique():
                einheitsname_not_found.append(name)

        df_relevant_parts = df_relevant_parts.reset_index(drop=True)
        #df_relevant_parts.loc[df_relevant_parts['Einheitsname'] == "Dummy", 'Einheitsname'] = 'Kein Einheitsname gefunden'

        df_relevant_parts = df_relevant_parts.drop_duplicates(subset=["Sachnummer"])

        df_relevant_parts = df_relevant_parts[df_relevant_parts['Einheitsname'] != 'Dummy'].reset_index(drop=True)

        df_relevant_parts = df_relevant_parts.sort_values(by=['Einheitsname'], ascending=True)

        logger.success("Output is prepared!")

        return df_preprocessed, df_relevant_parts, einheitsname_not_found, ncar

# %%
def main():
    
    data_path = Path(config["test_file_path"])
    df, ncar = read_file(data_path, raw=True)

    df_preprocessed, df_relevant_parts, einheitsname_not_found, ncar = Identifier.classification_on_new_data(df)
    print(df_relevant_parts)
    print(einheitsname_not_found)

# %%
if __name__ == "__main__":
    
    main()