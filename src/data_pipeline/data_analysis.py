import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from feature_engineering import transform_boundingbox, find_valid_space
from config import paths
from loguru import logger
import os
from datetime import datetime

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

def plot_bounding_boxes_one_vehicle(data_path):
    df = pd.read_excel(data_path, index_col=0) 
    df = df[(df['X-Max'] != 0) & (df['X-Min'] != 0)]
    df = df[df["Relevant fuer Messung"] == "Ja"]
    unique_names = df["Einheitsname"].unique().tolist()
    unique_names.sort()
    for name in unique_names:
        print(name)
        df_new = df[(df["Einheitsname"] == name)]
        plot_vehicle(df_new, add_valid_space=True, preprocessed_data=False, mirrored=False)  

def plot_bounding_boxes_all_vehicle_by_name(data_path):
    df_preprocessed = pd.read_excel(data_path,index_col=0) 
    # Hier muss noch wasgeändert werden
    #df_preprocessed, df_for_plot = preprocess_dataset(df)
    #plot_vehicle(df_for_plot, add_valid_space=True, preprocessed_data=False, mirrored=False)

def analyse_data_split(df_preprocessed, y_train, y_val, y_test, model_folder_path, binary_model):
    
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

def store_class_distribution(df, class_column, storage_path):
    # Verteilung der Klassen ermitteln
    class_counts = df[class_column].value_counts()

    # Balkendiagramm erstellen
    plt.figure(figsize=(10, 10))
    class_counts.plot(kind='bar')
    plt.xlabel(class_column)
    plt.ylabel('Number of Car Parts')
    plt.title('Class Distribution')
    plt.savefig(storage_path + f'/Distribution_{class_column}_{len(class_counts)}.png', dpi=150)