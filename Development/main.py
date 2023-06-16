import pandas as pd

from pathlib import Path
from loguru import logger

from Data_Preprocessing import prepare_and_add_labels, load_csv_into_df
from LightGBM_Relevanz import train_lgbm_binary_model
from LightGBM_Einheitsnamen import train_lgbm_multiclass_model
from model_predictions import predict_on_new_data
from plot_functions import plot_vehicle
from Data_Preprocessing import preprocess_dataset

# %%
def main():
    train_lgbm_relevance_model = False
    train_lgbm_name_model = False
    prediction_on_new_data = False
    plot_bounding_boxes_one_vehicle = False
    plot_bounding_boxes_all_vehicle_by_name = False

    dataset_path = ""

    if train_lgbm_relevance_model:
        train_lgbm_binary_model()

    if train_lgbm_name_model:
        train_lgbm_multiclass_model()

    if prediction_on_new_data:
        data_path = Path("../data/original_data_new")
        dataframes = load_csv_into_df(data_path, original_prisma_data=False, move_to_archive=False)
        df, ncars = prepare_and_add_labels(dataframes, save_as_excel=False)   
        df_predicted, einheitsname_not_found, ncars = predict_on_new_data(df[0])

        df_predicted.to_excel(f"../data/predicted/{ncars[0]}_relevante_Bauteile.xlsx")

        logger.info("The following car parts are not found in the data:\n",einheitsname_not_found)
        logger.success(f"The prediction is done and the result is stored here: ../data/predicted/{ncars[0]}_relevante_Bauteile.xlsx!")

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