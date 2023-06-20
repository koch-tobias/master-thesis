import pandas as pd

from pathlib import Path
from loguru import logger

from Data_Preprocessing import load_csv_into_df
from LightGBM_Binary import train_lgbm_binary_model
from LightGBM_Multiclass import train_lgbm_multiclass_model
from model_predictions import predict_on_new_data
from plot_functions import plot_vehicle
from Data_Preprocessing import preprocess_dataset

# %%
def main():
    train_lgbm_relevance_model = False
    train_lgbm_name_model = False
    label_new_data = True
    plot_bounding_boxes_one_vehicle = False
    plot_bounding_boxes_all_vehicle_by_name = False
    dataset_path = "Path which dataset should be plotted"

    if train_lgbm_relevance_model:
        train_lgbm_binary_model()

    if train_lgbm_name_model:
        train_lgbm_multiclass_model()

    if label_new_data:
        dataframes = load_csv_into_df(original_prisma_data=True, label_new_data=True)
        for df in dataframes:
            df_with_label_columns, df_relevant_parts, einheitsname_not_found, ncar = predict_on_new_data(df)

            for index, row in df_relevant_parts.iterrows():
                sachnummer = row['Sachnummer']
                einheitsname = row['Einheitsname']
                
                if sachnummer in df['Sachnummer'].values:
                    df_with_label_columns.loc[df_with_label_columns['Sachnummer'] == sachnummer, 'Relevant fuer Messung'] = "Ja"
                    df_with_label_columns.loc[df_with_label_columns['Sachnummer'] == sachnummer, 'Einheitsname'] = einheitsname

            df_with_label_columns.to_excel(f"data/pre_labeled_data/{ncar}_labeled_test.xlsx")

            logger.info(f"The following car parts are not found in the data: {einheitsname_not_found}")
            logger.success(f"The prediction is done and the result is stored here: data/pre_labeled_data/{ncar}_labeled.xlsx!")

            logger.info('__________________________________________________________________________________________')

    if plot_bounding_boxes_one_vehicle:
        df = pd.read_excel(dataset_path, index_col=0) 
        df = df[(df['X-Max'] != 0) & (df['X-Min'] != 0)]
        df = df[df["Relevant fuer Messung"] == "Ja"]
        unique_names = df["Einheitsname"].unique().tolist()
        unique_names.sort()
        for name in unique_names:
            print(name)
            df_new = df[(df["Einheitsname"] == name)]
            plot_vehicle(df_new, add_valid_space=True, preprocessed_data=False, mirrored=False)

    if plot_bounding_boxes_all_vehicle_by_name:
        df = pd.read_excel(dataset_path,index_col=0) 
        df_preprocessed, df_for_plot = preprocess_dataset(df, cut_percent_of_front=0.20)
        plot_vehicle(df_for_plot, add_valid_space=True, preprocessed_data=False, mirrored=False)
    



# %%
if __name__ == "__main__":
    
    main()