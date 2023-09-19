from loguru import logger
import os
import yaml
from yaml.loader import SafeLoader
import sys
sys.path.append(os.getcwd())

from src.utils import load_data_into_df
from src.deployment.inference import Identifier


with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

def label_data() -> None:
    ''' 
    Label the data by predicting relevant car parts and unique names on the new dataset. It saves the labeled data as a CSV file in the specified folder. 
    If unique names are not found, it asks to add them manually.
    Args: None
    Return: None 
    '''
    dataframes, ncar = load_data_into_df(raw=True)
    for df in dataframes:
        df_with_label_columns, df_relevant_parts, einheitsname_not_found, ncar = Identifier.classification_on_new_data(df)

        # Generate the prelabeled data
        for index, row in df_relevant_parts.iterrows():
            label_column_binary = config['labels']['binary_column']
            label_column_multiclass = config['labels']['multiclass_column']
            sachnummer = row['Sachnummer']
            einheitsname = row[label_column_multiclass]
            
            if sachnummer in df['Sachnummer'].values:
                df_with_label_columns.loc[df_with_label_columns['Sachnummer'] == sachnummer, label_column_binary] = config['labels']['binary_label_1']
                df_with_label_columns.loc[df_with_label_columns['Sachnummer'] == sachnummer, label_column_multiclass] = einheitsname

        features = config["relevant_features"] + [label_column_binary, label_column_multiclass]
        df_with_label_columns = df_with_label_columns[features]
        df_with_label_columns.to_csv(f"data/pre_labeled/{ncar}_labeled.csv")

        logger.info(f"The following car parts are not found in your dataset: {einheitsname_not_found} If essential, please add this car parts manually!")
        logger.success(f"The prediction is done and the result is stored here: data/pre_labeled_data/{ncar}_labeled.csv!")

# %%
def main():
    label_data()

# %%
if __name__ == "__main__":
    
    main()