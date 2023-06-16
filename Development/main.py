# %%
from pathlib import Path
from loguru import logger
import pickle
import numpy as np

from Data_Preprocessing import preprocess_dataset, prepData_Preprocessingre_and_add_labels, load_csv_into_df, prepare_text

# %%
def main():
    # Define the path to the folder containing the data (xls files)
    data_path = Path("../data/original_data_new")
    dataframes = load_csv_into_df(data_path, original_prisma_data=False, move_to_archive=False)
    df, ncars = prepare_and_add_labels(dataframes, save_as_excel=False)
    
    # Load model
    model_path = "../models/lgbm_15052023_1541.pkl"
    with open(model_path, "rb") as fid:
        lgbm = pickle.load(fid)

    # Load the vectorizer from the file
    vectorizer_path = "../models/vectorizer_15052023_1541.pkl"
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    # Get the vocabulary of the training data
    vocab_path = '../models/vocabulary_15052023_1541.pkl'
    with open(vocab_path, 'rb') as f:
        vocabulary = pickle.load(f)
    
    for i in range(len(df)):

        df_preprocessed, df_for_plot = preprocess_dataset(df[i], cut_percent_of_front=0.25)

        logger.info(f"Start predicting relevant parts for the {ncars[i]} ...")

        df_preprocessed["Benennung (dt)"] = df_preprocessed.apply(lambda x: prepare_text(x["Benennung (dt)"]), axis=1)

        # Convert the vocabulary list to a dictionary
        vocabulary_dict = {word: index for index, word in enumerate(vocabulary)}

        # Set the vocabulary of the vectorizer to the loaded vocabulary
        vectorizer.vocabulary_ = vocabulary_dict
        X = vectorizer.transform(df_preprocessed['Benennung (dt)']).toarray()

        # Combine text features with other features
        #X = np.concatenate((X, df_preprocessed[['center_x', 'center_y', 'center_z','length','width','height','theta_x','theta_y','theta_z']].values), axis=1)

        y_pred = lgbm.predict(X)
        y_pred = np.round(y_pred)

        for index, row in df_preprocessed.iterrows():
            if y_pred[index] == 1: 
                df_preprocessed.loc[index,'Relevant fuer Messung'] = 'Ja'
            else:
                df_preprocessed.loc[index,'Relevant fuer Messung'] = 'Nein'

        df_preprocessed = df_preprocessed[df_preprocessed['Relevant fuer Messung'] == 'Ja']

        df_preprocessed = df_preprocessed.loc[:,["Sachnummer", "Benennung (dt)", "Relevant fuer Messung", "Einheitsname"]]

        df_preprocessed.to_excel(f"../data/predicted/{ncars[i]}_relevante_Bauteile.xlsx")

        logger.success(f"The prediction is done and the result is stored here: data/predicted/{ncars[i]}_labeled_test.xlsx!")

        logger.info('__________________________________________________________________________________________')

# %%
if __name__ == "__main__":
    
    main()