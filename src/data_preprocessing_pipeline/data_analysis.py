import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from loguru import logger
import os

import yaml
from yaml.loader import SafeLoader
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

import sys
sys.path.append(config['paths']['project_path'])

def analyse_data_split(df_preprocessed: pd.DataFrame, y_train: np.array, y_val: np.array, y_test: np.array, model_folder_path: str, binary_model: bool) -> None:
    '''
    Create plots for the class distribution
    Args:
        df_preprocessed: dataframe of the preprocessed data
        y_train: class labels
        y_val: class labels
        y_test: class labels
        model_folder_path: path where the plot should be stored
        binary_model: 
    Return:
        None
    '''
    
    logger.info("Start analysing the preprocessed dataset...")

    width = 0.25
    x_labels = ['Training', 'Validation', 'Test']
    if binary_model:
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.rcParams.update({'font.size': 12})
        y_not_relevant = [np.count_nonzero(y_train == 0), np.count_nonzero(y_val == 0), np.count_nonzero(y_test == 0)]
        y_relevant = [np.count_nonzero(y_train == 1), np.count_nonzero(y_val == 1), np.count_nonzero(y_test == 1)]
        ax.set_ylabel('Number of Car Parts')
        ax.set_xlabel('Datasets')
        ax.set_title('Training, Validation and Test Split')
        ax.bar(x_labels, y_not_relevant, width, color='teal')
        ax.bar(x_labels, y_relevant, width, color='lightseagreen')
        ax.legend(labels=['Not Relevant', 'Relevant'])
        fig.savefig(model_folder_path + 'binary/Binary_train_val_test_split.png', dpi=150)
    else:
        class_names = df_preprocessed[config['labels']['multiclass_column']].unique()
        class_names = sorted(class_names)
        datasets_name = ["Train", "Validation", "Test"]
        datasets = [y_train, y_val, y_test]
        df_count_unique_names = pd.DataFrame(columns=class_names)
        for skip_dummy in range(2):
            fig, ax = plt.subplots(figsize=(40, 10))
            plt.rcParams.update({'font.size': 12})

            for i in range(len(datasets_name)):
                df_count_unique_names.loc[i,"Dataset"] = datasets_name[i]
                n = 0
                for name in class_names:
                    if skip_dummy == 0:
                        df_count_unique_names.loc[i, name] = np.count_nonzero(datasets[i] == n)
                    else:
                        if name == "Dummy":
                            df_count_unique_names.loc[i, name] = 0
                        else:
                            df_count_unique_names.loc[i, name] = np.count_nonzero(datasets[i] == n)
                    n = n + 1
            df_count_unique_names.plot(x='Dataset', kind='bar', stacked=True, align='center', title='Training, Validation and Test Split', colormap='tab20b', ax=ax)
            ax.legend(bbox_to_anchor=(1.0,0.5), loc='center left')
            ax.set_xticklabels(df_count_unique_names['Dataset'], rotation = 'horizontal')
            ax.set_ylabel('Number of Car Parts')
            ax.set_xlabel('Dataset')
            fig.tight_layout()

            if skip_dummy == 0:
                fig.savefig(model_folder_path + 'multiclass/Multiclass_train_val_test_split.png', dpi=150)
            else: 
                fig.savefig(model_folder_path + 'multiclass/Multiclass_train_val_test_split_without_dummy.png', dpi=150)
    logger.success("Dataset analysed!")

def store_class_distribution(df: pd.DataFrame, class_column: list, storage_path: str) -> None:
    '''
    Create a bar plot with the class label distribution and save the plot in the storage_path
    Args:
        df = dataframe of the labeled data
        class_colums = list of class labels
        storage_path = path where the plot should be stored
    Return: None
    '''
    # Verteilung der Klassen ermitteln
    class_counts = df[class_column].value_counts()

    # Balkendiagramm erstellen
    plt.figure(figsize=(10, 10))
    class_counts.plot(kind='bar')
    plt.xlabel(class_column)
    plt.ylabel('Number of Car Parts')
    plt.title('Class Distribution')
    plt.savefig(storage_path + f'Distribution_{class_column}_{len(class_counts)}.png', dpi=150)

def store_feature_distribution(df, storage_path: str):
    '''
    Create a distribution plot for each numeric feature and save the plot in the storage_path
    Args:
        df = dataframe of the labeled data
        storage_path = path where the plot should be stored
    Return: None
    '''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerics_columns = list(df.select_dtypes(include=numerics).columns)
    plt. close('all')
    os.makedirs(storage_path + "Feature_distributions")
    for col in numerics_columns:
        plt.figure(figsize=(8,6))
        plt.hist(df[col], bins=30)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {col}")
        plt.savefig(storage_path + f"Feature_distributions/Distribution_of_{col}.png")
        plt. close('all')