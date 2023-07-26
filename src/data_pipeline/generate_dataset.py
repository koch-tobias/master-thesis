from preprocessing import preprocess_dataset, load_csv_into_df, combine_dataframes
from data_analysis import store_class_distribution
from augmentation import data_augmentation
from src.config import general_params, paths

from datetime import datetime
import os

# %%
def main():
    dateTimeObj = datetime.now()
    timestamp = dateTimeObj.strftime("%d%m%Y_%H%M")
    storage_path = paths["processed_dataset"] + "/" + timestamp

    dataframes_list, ncars = load_csv_into_df(original_prisma_data=False, label_new_data=False)

    df_combined = combine_dataframes(dataframes_list)
    df_preprocessed, df_for_plot = preprocess_dataset(df_combined)

    if general_params["augmentation"]:
        # Generate the new dataset
        df_preprocessed = data_augmentation(df_preprocessed)

    os.makedirs(storage_path)
    df_preprocessed.to_csv(storage_path + "/processed_dataset.csv")

    store_class_distribution(df_preprocessed, "Relevant fuer Messung", storage_path)
    store_class_distribution(df_preprocessed, "Einheitsname", storage_path)
    filtered_df = df_preprocessed[df_preprocessed["Einheitsname"] != "Dummy"]
    store_class_distribution(filtered_df, "Einheitsname", storage_path)


# %%
if __name__ == "__main__":
    
    main()