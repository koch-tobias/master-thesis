import pandas as pd
import numpy as np

import pickle
import os
from loguru import logger

import yaml
from yaml.loader import SafeLoader
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

import sys
sys.path.append(config['paths']['project_path'])
from src.data_preparation_pipeline.data_preparation import Preperator
from src.data_preprocessing_pipeline.feature_engineering import Feature_Engineering
from src.data_preprocessing_pipeline.data_cleaning import DataCleaner
from src.utils import read_file

class Identifier():

    @staticmethod
    def search_in_logging(text:str, model_folder_path: str) -> str or None:
        ''' 
        Description: Returns the value after a searched text by reading the 'logging.txt' file of the given model path.
        Args:
            text: string which should be searched in the logging file
            model_folder_path: path of folder containing the model
        Return:
            value after the searched string
            None if the string is not found in the 'logging.txt' file 
        '''
        logging_path = model_folder_path + "/logging.txt"
        with open(logging_path, 'r') as file:
            for line in file:
                if text in line:
                    return line.split(":")[1].strip()
        return None

    @staticmethod
    def get_model(folder_path: str):
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
        final_model_path = folder_path + "/final_model.pkl"
        pretrained_model_path = folder_path + "/model.pkl"

        if os.path.exists(final_model_path):
            model_path =  final_model_path
        else:
            model_path =  pretrained_model_path

        with open(model_path, "rb") as fid:
            model = pickle.load(fid)

        # Get the dataset path from the logging file
        dataset_path = Identifier.search_in_logging(text="Dataset:", model_folder_path=folder_path)

        # Load the vectorizer from the file
        vectorizer_path = dataset_path + "vectorizer.pkl"
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

        # Get the vocabulary of the training data
        vocab_path = dataset_path + "vocabulary.pkl"
        with open(vocab_path, 'rb') as f:
            vocabulary = pickle.load(f) 

        # Get the used boundingbox features
        bbox_features_path = dataset_path + "boundingbox_features.pkl"
        with open(bbox_features_path, 'rb') as f:
            bbox_features = pickle.load(f)  

        return model, vectorizer, vocabulary, bbox_features

    @staticmethod
    def get_X(vocab, vectorizer, bbox_features: list, df_preprocessed: pd.DataFrame, model_folder_path: str) -> np.array:
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
            best_iteration = model.get_best_iteration() - 1

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

    # %%
    @staticmethod
    def store_predictions(y_test: np.array, y_pred: np.array, probs: np.array, df_preprocessed: pd.DataFrame, df_test: pd.DataFrame, model_folder_path: str, binary_model: bool) -> None:
        ''' 
        Stores wrong predictions, true labels, predicted labels, probabilities, and additional information based on the input parameters in a CSV file.
        Args:
            y_test: true labels
            y_pred: predicted labels
            probs: probability values
            df_preprocessed: preprocessed DataFrame
            df_test: test DataFrame
            model_folder_path: path to store the wrong_predictions.csv file
            binary_model: bool, indicates whether binary classification or multiclass classification is expected
        Return: None 
        '''
        if binary_model:
            class_names = df_preprocessed['Relevant fuer Messung'].unique()
        else:
            class_names = df_preprocessed["Einheitsname"].unique()
            class_names = sorted(class_names)

        df_wrong_predictions = pd.DataFrame(columns=['Sachnummer', 'Benennung (dt)', 'Derivat', 'Predicted', 'True', 'Probability'])

        try:
            y_test = y_test.to_numpy()
        except:
            pass
        
        # Store all wrongly identified car parts in a csv file to analyse the model predictions
        for i in range(len(y_test)):
            if y_pred[i] != y_test[i]:
                df_wrong_predictions.loc[i,"Sachnummer"] = df_test.loc[i, "Sachnummer"]
                df_wrong_predictions.loc[i,"Benennung (dt)"] = df_test.loc[i, "Benennung (dt)"]
                df_wrong_predictions.loc[i,"Derivat"] = df_test.loc[i, "Derivat"]
                df_wrong_predictions.loc[i,"Predicted"] = class_names[y_pred[i]]
                df_wrong_predictions.loc[i,"True"] = class_names[y_test[i]]
                if binary_model:
                    if probs[i][1] >= config["prediction_settings"]["prediction_threshold"]:
                        df_wrong_predictions.loc[i,"Probability"] = probs[i][1]
                    else:
                        df_wrong_predictions.loc[i,"Probability"] = 1 - probs[i][1]
                else:
                    df_wrong_predictions.loc[i,"Probability"] = probs[i][1]
    
        df_wrong_predictions.to_csv(model_folder_path + "wrong_predictions.csv")

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

        logger.info("Preprocess data...")
        df_new_features = Feature_Engineering.add_new_features(df)
        df_preprocessed, df_for_plot = DataCleaner.clean_dataset(df_new_features)
        logger.info("Dataset successfully preprocessed!")

        logger.info("Load pretrained models...")
        model_folder_path_binary = "final_models/Binary_model"
        model_folder_path_multiclass = "final_models/Multiclass_model"
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

        # Search in logging which dataset is used for training
        dataset_path_binary = Identifier.search_in_logging(text="Dataset:", model_folder_path=model_folder_path_binary)
        dataset_path_multiclass = Identifier.search_in_logging(text="Dataset:", model_folder_path=model_folder_path_multiclass)

        # Load the LabelEncoder
        with open(dataset_path_multiclass + 'label_encoder.pkl', 'rb') as f:
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
        
        '''
        logger.info("Load dataset used for training..")
        trainset = pd.read_csv(dataset_path_binary + "processed_dataset.csv")
        trainset_relevant_parts = trainset[trainset["Relevant fuer Messung"] == "Ja"]
        trainset_relevant_parts = trainset_relevant_parts[(trainset_relevant_parts['X-Min_transf'] != 0) & (trainset_relevant_parts['X-Max_transf'] != 0)]    
        unique_names = trainset_relevant_parts["Einheitsname"].unique().tolist()
        unique_names.sort()
        logger.success("Dataset loaded!")
        
        logger.info("Valid identified car parts by comparing the position of the new car part to them in the trainset")
        for index, row in df_relevant_parts.iterrows():
            for name in unique_names:
                trainset_name = trainset_relevant_parts[(trainset_relevant_parts["Einheitsname"] == name)].reset_index(drop=True)
                corners, _, _, _ = Feature_Engineering.find_valid_space(df=trainset_name)
                x_min = np.min(corners[:, 0])
                x_max = np.max(corners[:, 0])
                y_min = np.min(corners[:, 1])
                y_max = np.max(corners[:, 1])
                z_min = np.min(corners[:, 2])
                z_max = np.max(corners[:, 2])
                valid_volume_min = trainset_name["volume"].min()
                valid_volume_max = trainset_name["volume"].max()
                
                if ((row["X-Min_transf"] == 0) and (row["X-Max_transf"] == 0)):
                    df_relevant_parts.loc[index,'In Bounding-Box-Position von'] = 'No Bounding-Box information'
                else:
                    df_relevant_parts.loc[index,'In Bounding-Box-Position von'] = 'None'
                    if ((row["X-Min_transf"] > x_min) and (row["X-Max_transf"] < x_max)):
                        if ((row["Y-Min_transf"] > y_min) and (row["Y-Max_transf"] < y_max)): 
                                if ((row["Z-Min_transf"] > z_min) and (row["Z-Max_transf"] < z_max)):
                                    if ((row["volume"] >= valid_volume_min*0.9) and (row["volume"] <= valid_volume_max*1.1)):
                                        df_relevant_parts.loc[index,'In Bounding-Box-Position von'] = name
                                        if (row["Wahrscheinlichkeit Relevanz"] > 0.95) and ((row["Einheitsname"] == "Dummy")):
                                            df_relevant_parts.loc[index,'Einheitsname'] = name
                                        break
        logger.success("Validation successfull!")
        '''
        logger.info("Prepare output...")
        # Load list of the uniform names (classes)
        with open(model_folder_path_binary + "list_of_uniform_names.pkl", 'rb') as names:
            uniform_names = pickle.load(names)
            
        # Check which uniform names are not identified
        einheitsname_not_found = []
        for name in uniform_names:        
            if name not in df_relevant_parts['Einheitsname'].unique():
                einheitsname_not_found.append(name)

        df_relevant_parts = df_relevant_parts.reset_index(drop=True)
        df_relevant_parts.loc[df_relevant_parts['Einheitsname'] == "Dummy", 'Einheitsname'] = 'Kein Einheitsname gefunden'

        df_relevant_parts = df_relevant_parts.drop_duplicates(subset=["Sachnummer"])

        logger.success("Output is prepared!")

        return df_preprocessed, df_relevant_parts, einheitsname_not_found, ncar

# %%
def main():
    
    data_path = config["paths"]["test_file_path"]
    df = read_file(data_path)

    df_preprocessed, df_relevant_parts, einheitsname_not_found, ncar = Identifier.classification_on_new_data(df)
    print(df_relevant_parts)
    print(einheitsname_not_found)

# %%
if __name__ == "__main__":
    
    main()