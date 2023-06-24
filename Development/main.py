import pandas as pd

from loguru import logger
from datetime import datetime

from Data_Preprocessing import load_csv_into_df
from LightGBM import train_lgbm_model
from model_predictions import predict_on_new_data
from plot_functions import plot_vehicle
from Data_Preprocessing import preprocess_dataset
from config import general_params

# %%
def main():
    train_lgbm_relevance_model = True
    train_lgbm_name_model = True
    label_new_data = False
    plot_bounding_boxes_one_vehicle = False
    plot_bounding_boxes_all_vehicle_by_name = False
    dataset_path_for_plot = "Path of dataset which should be plotted"

    dateTimeObj = datetime.now()
    timestamp = dateTimeObj.strftime("%d%m%Y_%H%M")

    folder_path = f"models/HyperparameterTuning_{timestamp}/"
    #folder_path = ""

    if train_lgbm_relevance_model:
        logger.info("Start training the binary models...")
        train_lgbm_model(folder_path, binary_model=True)

    if train_lgbm_name_model:
        logger.info("Start training the multiclass models...")
        train_lgbm_model(folder_path, binary_model=False)

    if label_new_data:
        dataframes = load_csv_into_df(original_prisma_data=True, label_new_data=True)
        for df in dataframes:
            df_with_label_columns, df_relevant_parts, einheitsname_not_found, ncar = predict_on_new_data(df, use_api=False)

            for index, row in df_relevant_parts.iterrows():
                sachnummer = row['Sachnummer']
                einheitsname = row['Einheitsname']
                
                if sachnummer in df['Sachnummer'].values:
                    df_with_label_columns.loc[df_with_label_columns['Sachnummer'] == sachnummer, 'Relevant fuer Messung'] = "Ja"
                    df_with_label_columns.loc[df_with_label_columns['Sachnummer'] == sachnummer, 'Einheitsname'] = einheitsname

            features = general_params["relevant_features"] + ['Relevant fuer Messung','Einheitsname']
            df_with_label_columns = df_with_label_columns[features]
            df_with_label_columns.to_excel(f"data/pre_labeled_data/{ncar}_labeled.xlsx")

            logger.info(f"The following car parts are not found in the data: {einheitsname_not_found}")
            logger.success(f"The prediction is done and the result is stored here: data/pre_labeled_data/{ncar}_labeled.xlsx!")

            logger.info('__________________________________________________________________________________________')

    if plot_bounding_boxes_one_vehicle:
        df = pd.read_excel(dataset_path_for_plot, index_col=0) 
        df = df[(df['X-Max'] != 0) & (df['X-Min'] != 0)]
        df = df[df["Relevant fuer Messung"] == "Ja"]
        unique_names = df["Einheitsname"].unique().tolist()
        unique_names.sort()
        for name in unique_names:
            print(name)
            df_new = df[(df["Einheitsname"] == name)]
            plot_vehicle(df_new, add_valid_space=True, preprocessed_data=False, mirrored=False)

    if plot_bounding_boxes_all_vehicle_by_name:
        df = pd.read_excel(dataset_path_for_plot,index_col=0) 
        df_preprocessed, df_for_plot = preprocess_dataset(df, cut_percent_of_front=0.20)
        plot_vehicle(df_for_plot, add_valid_space=True, preprocessed_data=False, mirrored=False)
    



# %%
if __name__ == "__main__":
    
    main()