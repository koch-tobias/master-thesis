from loguru import logger

from data_pipeline.preprocessing import load_csv_into_df
from deployment_pipeline.prediction import predict_on_new_data
from config import general_params

# %%
def main():
    
    dataframes, ncars = load_csv_into_df(original_prisma_data=True, label_new_data=True)
    for df in dataframes:
        df_with_label_columns, df_relevant_parts, einheitsname_not_found, ncar = predict_on_new_data(df)

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

# %%
if __name__ == "__main__":
    
    main()