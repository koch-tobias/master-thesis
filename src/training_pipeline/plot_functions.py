import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.data_pipeline.feature_engineering import transform_boundingbox, find_valid_space
from sklearn.metrics import ConfusionMatrixDisplay
import lightgbm as lgb
import pickle
from loguru import logger


import yaml
from yaml.loader import SafeLoader
with open('../config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# %%
def plot_bounding_box(ax, transformed_boundingbox: np.array, designation: str, label_relevant: str):
    '''
    This function is used to plot a three-dimensional bounding box.
    Args:
        ax: The three-dimensional axis to plot the bounding box.
        transformed_boundingbox: A numpy array which contains the transformation matrix to transform the points of the bounding box.
        designation: A string value representing the designation of the bounding box.
        label_relevant: A string value representing the label of the bounding box relative to the object it bounds. ('Ja' for relevant bounding box, 'Nein' for irrelevant bounding box, and Unentschieden for an undefined bounding box)
    Returns:
        relevant_count: an integer value representing the count of relevant bounding boxes.
    '''
    # Define the edges of the bounding box
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]

    # Plot the edges of the bounding box
    for edge in edges:
        if label_relevant == "Nein":
            ax.plot(transformed_boundingbox[edge, 0], transformed_boundingbox[edge, 1], transformed_boundingbox[edge, 2], color='#999999', alpha=0.5, ms=10)
            relevant_count = 0
        elif label_relevant == "Ja":
            ax.plot(transformed_boundingbox[edge, 0], transformed_boundingbox[edge, 1], transformed_boundingbox[edge, 2], 'r-')    
            #ax.text(transformed_boundingbox[0, 0], transformed_boundingbox[0, 1], transformed_boundingbox[0, 2], designation, color='black', fontsize=10)
            relevant_count = 1
        else:
            ax.plot(transformed_boundingbox[edge, 0], transformed_boundingbox[edge, 1], transformed_boundingbox[edge, 2], 'g--')   
            relevant_count = 0

    return relevant_count

# %%
def plot_vehicle(df: pd.DataFrame, add_valid_space: bool, preprocessed_data: bool, mirrored: bool):
    ''' 
    This function takes a Pandas dataframe containing information about the bounding boxes of a vehicle and plot them in a 3D space. The plot can be mirrored and can have a valid space bounding box added. 
    It iterates through the dataframe, transforms the bounding box values and plots each. It also counts the relevant parts found and not found, and prints the result. 
    Args:
        df: Pandas dataframe containing information about the bounding boxes of a vehicle.
        add_valid_space: boolean variable indicating if a valid space bounding box should be added to the plot.
        preprocessed_data: boolean variable indicating if the data has already been preprocessed or not.
        mirrored: boolean variable indicating if the plot should be mirrored or not. 
    Return: None. 
    '''
    fig = plt.figure(figsize=(10, 20), dpi=100)

    ax = fig.add_subplot(111, projection='3d')

    # Iterate through the dataframe and plot each bounding box
    count_relevant_parts = 0
    count_all = 0

    for index, row in df.iterrows():
            if preprocessed_data:
                    transformed_boundingbox = np.array([[row['X-Min_transf'], row['Y-Min_transf'], row['Z-Min_transf']],
                                            [row['X-Min_transf'], row['Y-Min_transf'], row['Z-Max_transf']],
                                            [row['X-Min_transf'], row['Y-Max_transf'], row['Z-Min_transf']],
                                            [row['X-Min_transf'], row['Y-Max_transf'], row['Z-Max_transf']],
                                            [row['X-Max_transf'], row['Y-Min_transf'], row['Z-Min_transf']],
                                            [row['X-Max_transf'], row['Y-Min_transf'], row['Z-Max_transf']],
                                            [row['X-Max_transf'], row['Y-Max_transf'], row['Z-Min_transf']],
                                            [row['X-Max_transf'], row['Y-Max_transf'], row['Z-Max_transf']]])
            else:    
                    transformed_boundingbox = transform_boundingbox(row['X-Min'], row['X-Max'], row['Y-Min'], row['Y-Max'], row['Z-Min'], row['Z-Max'], row['ox'], row['oy'], row['oz'], row['xx'], row['xy'], row['xz'], row['yx'], row['yy'], row['yz'], row['zx'], row['zy'], row['zz'])
            relevant = plot_bounding_box(ax, transformed_boundingbox, row['Benennung (dt)'], row["Relevant fuer Messung"])
            count_relevant_parts = count_relevant_parts + relevant
            count_all = count_all + 1

    if add_valid_space:
            corners = find_valid_space(df.reset_index(drop=True))
            plot_bounding_box(ax, corners, 'Valid Space', 'space')

    print(f"{count_relevant_parts} relevant parts found")
    print(f"Still {count_all-count_relevant_parts} not relevant parts found")

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set axis limits
    ax.set_xlim(-2000, 5000)
    ax.set_ylim(-1500, 1500)
    ax.set_zlim(-100, 1500)
    
    ax.set_aspect('equal', adjustable='box')

    if mirrored:
            ax.invert_xaxis()

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig("../plots_images/bounding_boxes/bounding_box_G65_step00.png", format='png', bbox_inches='tight', pad_inches=0)

    # Show the plot
    plt.show()    
    plt.close()

def analyse_data(df_preprocessed: pd.DataFrame, y_train: np.array, y_val: np.array, y_test: np.array, model_folder_path: str, binary_model: bool):
    ''' 
    This function analyses a preprocessed dataset by plotting the distribution of car parts among the Training, Validation and Test datasets. 
    If binary_model is True, it plots the distribution of relevant and not relevant car parts. Otherwise, it plots the distribution of all car parts by class name. 
    It saves these plots in a specified folder. 
    Args:
        df_preprocessed: Pandas dataframe containing the preprocessed dataset.
        y_train: numpy array containing the target values for the training dataset.
        y_val: numpy array containing the target values for the validation dataset.
        y_test: numpy array containing the target values for the test dataset.
        model_folder_path: string containing the path where the resulting plots should be saved.
        binary_model: boolean variable indicating whether the model is binary (relevant/not relevant) or not (multiclass). 
    Return: None. 
    '''
    
    logger.info("Start analysing the preprocessed dataset...")
    # Analyze the dataset
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
        fig.savefig(model_folder_path + 'Binary_train_val_test_split.png', dpi=150)
    else:
        class_names = df_preprocessed['Einheitsname'].unique()
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
                fig.savefig(model_folder_path + 'Multiclass_train_val_test_split.png', dpi=150)
            else: 
                fig.savefig(model_folder_path + 'Multiclass_train_val_test_split_without_dummy.png', dpi=150)
    plt.close('all')
    logger.success("Dataset analysed!")

def plot_metric_custom(evals: dict, best_iteration: int, model_folder_path: str, method: str, binary: bool, finalmodel: bool):
    ''' 
    This function plots the Training and Validation AUC and Loss of a machine learning model, given the evaluation information of the model. 
    It saves the plots in a specified folder. The function also checks whether the model is a binary or multiclass model and whether it is the final model or not, in order to add an appropriate prefix to the resulting plot file names. 
    Args:
        evals: dictionary containing the evaluation results of the model.
        best_iteration: integer indicating the iteration where early stopping occurred.
        model_folder_path: string containing the path where the resulting plots should be saved.
        method: string indicating the machine learning method used (either 'catboost' or 'xgboost').
        binary: boolean variable indicating whether the model is binary or not.
        finalmodel: boolean variable indicating whether the model is the final model or not. 
    Return: None 
    '''
    if method == 'catboost':
        if binary:
            metric_0 = config["cb_params_binary"]["metrics"][0]
            metric_1 = config["cb_params_binary"]["metrics"][1]
        else:
            if config["cb_params_multiclass"]["metrics"][0] == 'AUC':
                metric_0 = 'AUC:type=Mu'
            else:
                metric_0 = config["cb_params_multiclass"]["metrics"][0]
            metric_1 = config["cb_params_multiclass"]["metrics"][1]             
    else:
        if binary:
            metric_0 = config["xgb_params_binary"]["metrics"][0]
            metric_1 = config["xgb_params_binary"]["metrics"][1]
        else:
            metric_0 = config["xgb_params_multiclass"]["metrics"][0]
            metric_1 = config["xgb_params_multiclass"]["metrics"][1] 

    if finalmodel:
         add_to_path = "final_model_"
    else:
         add_to_path = ""

    val_auc = evals["validation_1"][metric_0]
    val_loss = evals["validation_1"][metric_1]
    train_auc = evals["validation_0"][metric_0]
    train_loss = evals["validation_0"][metric_1]

    plt.rcParams["figure.figsize"] = (10, 10)
    plt.plot(train_auc, label='Train AUC')
    plt.plot(val_auc, label='Validation AUC')
    plt.xlabel('Number of Iterations')
    plt.ylabel('AUC')
    plt.title('Training and Validation AUC')
    plt.axvline(best_iteration, color='b', label = 'early stopping')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(model_folder_path + add_to_path + 'auc_plot.png')
    plt.close()

    plt.rcParams["figure.figsize"] = (10, 10)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.axvline(best_iteration, color='b', label = 'early stopping')
    plt.legend()
    plt.ylim([-0.5, 4])
    plt.savefig(model_folder_path + add_to_path + 'loss_plot.png')
    plt.close()

def store_metrics(evals: dict, best_iteration: int, model_folder_path: str, binary_model: bool, finalmodel: bool):
    ''' 
    This function plots the Training and Validation Loss of a machine learning model, given the evaluation information of the model. 
    It saves the plots in a specified folder. The function also checks whether the model is a binary or multiclass model and whether it is the final model or not, in order to add an appropriate prefix to the resulting plot file names. 
    Args:
        evals: dictionary containing the evaluation results of the model.
        best_iteration: integer indicating the iteration where early stopping occurred.
        model_folder_path: string containing the path where the resulting plots should be saved.
        binary_model: boolean variable indicating whether the model is binary or not.
        finalmodel: boolean variable indicating whether the model is the final model or not. 
    Return: None. 
    '''

    if finalmodel:
         add_to_path = "final_model_"
    else:
         add_to_path = ""

    if binary_model:
        plt.rcParams["figure.figsize"] = (10, 10)
        lgb.plot_metric(evals, metric=config["lgbm_params_binary"]["metrics"][1])
        plt.title("")
        plt.xlabel('Iterationen', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(['Training', 'Validation'], fontsize=12)
        plt.axvline(best_iteration, color='b', label = 'early stopping')
        plt.ylim([-0.5, 4])
        plt.savefig(model_folder_path + add_to_path + 'binary_logloss_plot.png')
        plt.close()

        plt.rcParams["figure.figsize"] = (10, 10)
        lgb.plot_metric(evals, metric=config["lgbm_params_binary"]["metrics"][0])
        plt.title("")
        plt.xlabel('Iterationen', fontsize=12 )
        plt.ylabel('AUC', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(['Training', 'Validation'], fontsize=12)
        plt.axvline(best_iteration, color='b', label = 'early stopping')
        plt.savefig(model_folder_path + add_to_path + 'auc_plot.png')
        plt.close()

    else:    
        plt.rcParams["figure.figsize"] = (10, 10)
        lgb.plot_metric(evals, metric=config["lgbm_params_multiclass"]["metrics"][1])
        plt.title("")
        plt.xlabel('Iterationen', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(['Training', 'Validation'], fontsize=12)
        plt.axvline(best_iteration, color='b', label = 'early stopping')
        plt.ylim([-0.5, 4])
        plt.savefig(model_folder_path + add_to_path + 'multi_logloss_plot.png')
        plt.close()

# %% 
def store_confusion_matrix(y_test: np.array, y_pred: np.array, folder_path: str, model_folder_path: str, binary_model: bool):
    ''' 
    This function stores the confusion matrix of a machine learning model, given the test labels and predicted labels.
    It saves the plot in a specified folder. The function also checks whether the model is binary or multiclass, in order to design the plot and the class names. If multiclass, it loads a label encoder from a saved file. 
    Args:
        y_test: numpy array containing the true labels of the test set.
        y_pred: numpy array containing the predicted labels of the test set.
        folder_path: string containing the path where the label_encoder.pkl file is stored.
        model_folder_path: string containing the path where the resulting plot should be saved.
        binary_model: boolean variable indicating whether the model is binary or not. 
    Return: None 
    '''
    if binary_model:
        class_names = ["Not relevant", "Relevant"]
        plt.rcParams["figure.figsize"] = (15, 15)
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, cmap='Blues', colorbar=False,  text_kw={'fontsize': 12})
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12 )
        plt.ylabel('True Label', fontsize=12)
        plt.savefig(model_folder_path + 'confusion_matrix.png')  

    else:
        with open(config["paths"]["folder_processed_dataset"] + 'label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)

        class_names = []
        classes_pred = le.inverse_transform(y_pred) 
        classes_true = le.inverse_transform(y_test)

        for name in le.classes_:
            if (name in classes_pred) or (name in classes_true):
                class_names.append(name)

        cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names)
        # Passe die Größe des Diagramms an
        fig, ax = plt.subplots(figsize=(40, 20))
        # Zeige die Konfusionsmatrix an
        cm_display.plot(ax=ax, xticks_rotation='vertical', cmap='Blues', colorbar=False,  text_kw={'fontsize': 12})
        # Speichere das Diagramm
        plt.savefig(model_folder_path + 'confusion_matrix.png')
    plt.close('all')