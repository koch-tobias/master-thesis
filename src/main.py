import pandas as pd

from loguru import logger
from datetime import datetime

from data.preprocessing import load_csv_into_df, preprocess_dataset
from models.train import train_model
from models.predict import predict_on_new_data
from visualization.plot_functions import plot_vehicle
from config_model import general_params, train_settings

# %%
def main():
    train_binary_model = True
    train_multiclass_model = True
    label_new_data = False
    plot_bounding_boxes_one_vehicle = False
    plot_bounding_boxes_all_vehicle_by_name = False

    method = train_settings["ml-method"]
    dataset_path_for_plot = "Path of dataset which should be plotted"

    dateTimeObj = datetime.now()
    timestamp = dateTimeObj.strftime("%d%m%Y_%H%M")

    folder_path = f"models/{method}_HyperparameterTuning_{timestamp}/"
    #folder_path = ""
    '''
    file_path = "C:/Users/q617269/Desktop/Masterarbeit_Tobias/repos/master-thesis/data/raw/G20_prismaexport-20230621-143916.xls"
    df = pd.read_excel(file_path, header=None, skiprows=1)
    df.columns = df.iloc[0]
    df = df.iloc[1:]    
    df_preprocessed, df_relevant_parts, einheitsname_not_found, ncar = predict_on_new_data(df, use_api=True)
    '''

    if train_binary_model:
        logger.info("Start training the binary models...")
        train_model(folder_path, binary_model=True, method=method)

    if train_multiclass_model:
        logger.info("Start training the multiclass models...")
        train_model(folder_path, binary_model=False, method=method)

    if label_new_data:
        dataframes, ncars = load_csv_into_df(original_prisma_data=True, label_new_data=True)
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
            df_with_label_columns.to_excel(f"data/pre_labeled/{ncar}_labeled.xlsx")

            logger.info(f"The following car parts are not found in your dataset: {einheitsname_not_found} If essential, please add this car parts manually!")
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
        df_preprocessed, df_for_plot = preprocess_dataset(df)
        plot_vehicle(df_for_plot, add_valid_space=True, preprocessed_data=False, mirrored=False)
    

# %%
if __name__ == "__main__":
    
    main()