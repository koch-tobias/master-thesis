from pathlib import Path
from loguru import logger

from Data_Preprocessing import prepare_and_add_labels, load_csv_into_df
from LightGBM_Relevanz import train_lgbm_binary_model
from LightGBM_Einheitsnamen import train_lgbm_multiclass_model
from model_predictions import predict_on_new_data

# %%
def main():
    train_lgbm_relevance_model = False
    train_lgbm_name_model = False
    prediction_on_new_data = False

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

# %%
if __name__ == "__main__":
    
    main()