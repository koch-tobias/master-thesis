import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data.boundingbox_calculations import transform_boundingbox, find_valid_space
from config_model import general_params, lgbm_params_binary, lgbm_params_multiclass
from sklearn.metrics import ConfusionMatrixDisplay
import lightgbm as lgb
import pickle
from loguru import logger

# %%
def plot_bounding_box(ax, transformed_boundingbox, designation, label_relevant):

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
def plot_vehicle(df: pd.DataFrame, add_valid_space:bool, preprocessed_data:bool, mirrored: bool):
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

def analyse_data(df_preprocessed, y_train, y_val, y_test, model_folder_path, binary_model):
    
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
    logger.success("Dataset analysed!")


def store_metrics(model, evals, model_folder_path, binary_model):

    early_stopping = model.best_iteration_-1
    if binary_model:
        plt.rcParams["figure.figsize"] = (10, 10)
        lgb.plot_metric(evals, metric=lgbm_params_binary["metrics"][1])
        plt.title("")
        plt.xlabel('Iterationen', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(['Training', 'Validation'], fontsize=12)
        plt.axvline(early_stopping, color='b', label = 'early stopping')
        plt.ylim([-0.5, 4])
        plt.savefig(model_folder_path + 'binary_logloss_plot.png')

        plt.rcParams["figure.figsize"] = (10, 10)
        lgb.plot_metric(evals, metric=lgbm_params_binary["metrics"][0])
        plt.title("")
        plt.xlabel('Iterationen', fontsize=12 )
        plt.ylabel('AUC', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(['Training', 'Validation'], fontsize=12)
        plt.axvline(early_stopping, color='b', label = 'early stopping')
        plt.savefig(model_folder_path + 'auc_plot.png')

    else:    
        plt.rcParams["figure.figsize"] = (10, 10)
        lgb.plot_metric(evals, metric=lgbm_params_multiclass["metrics"][1])
        plt.title("")
        plt.xlabel('Iterationen', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(['Training', 'Validation'], fontsize=12)
        plt.axvline(early_stopping, color='b', label = 'early stopping')
        plt.ylim([-0.5, 4])
        plt.savefig(model_folder_path + 'multi_logloss_plot.png')

# %% 
def store_confusion_matrix(y_test, y_pred, model_folder_path, binary_model):

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
        with open(model_folder_path + 'label_encoder.pkl', 'rb') as f:
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