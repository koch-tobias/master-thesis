from preprocessing import preprocess_dataset, load_csv_into_df, combine_dataframes
from data_analysis import store_class_distribution
from augmentation import data_augmentation
from src.utils import train_test_val
from src.config import general_params

from src.data_pipeline.data_analysis import analyse_data_split

from datetime import datetime
import os
import pickle
from loguru import logger

def generate_dataset_dict(df, storage_path, binary_model):

    X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test, weight_factor = train_test_val(df, model_folder_path=storage_path, binary_model=binary_model)

    analyse_data_split(df, y_train, y_val, y_test, storage_path, binary_model) 
    
    train_val_test_dict = dict({
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "weight_factor": weight_factor
    })

    train_val_test_dataframes = dict({
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test
    })

    if binary_model:
        with open(storage_path + 'binary_train_test_val_split.pkl', 'wb') as handle:
            pickle.dump(train_val_test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(storage_path + 'binary_train_test_val_dataframes.pkl', 'wb') as handle:
            pickle.dump(train_val_test_dataframes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(storage_path + 'multiclass_train_test_val_split.pkl', 'wb') as handle:
            pickle.dump(train_val_test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(storage_path + 'multiclass_train_test_val_dataframes.pkl', 'wb') as handle:
            pickle.dump(train_val_test_dataframes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.success("Splitted datasets are successfully stored!")

# %%
def main():
    dateTimeObj = datetime.now()
    timestamp = dateTimeObj.strftime("%d%m%Y_%H%M")
    storage_path = f"data/processed/{timestamp}/"

    dataframes_list, ncars = load_csv_into_df(original_prisma_data=False, label_new_data=False)

    df_combined = combine_dataframes(dataframes_list)
    df_preprocessed, df_for_plot = preprocess_dataset(df_combined)

    if general_params["augmentation"]:
        # Generate the new dataset
        df_preprocessed = data_augmentation(df_preprocessed)

    os.makedirs(storage_path)
    df_preprocessed.to_csv(storage_path + "processed_dataset.csv")

    generate_dataset_dict(df_preprocessed, storage_path, binary_model=True)
    generate_dataset_dict(df_preprocessed, storage_path, binary_model=False)

    logger.info("Generate and store the class distribution plots...")
    store_class_distribution(df_preprocessed, "Relevant fuer Messung", storage_path)
    store_class_distribution(df_preprocessed, "Einheitsname", storage_path)
    filtered_df = df_preprocessed[df_preprocessed["Einheitsname"] != "Dummy"]
    store_class_distribution(filtered_df, "Einheitsname", storage_path)
    logger.success("Plots successfully stored!")


# %%
if __name__ == "__main__":
    
    main()