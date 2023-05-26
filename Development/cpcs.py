import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
from ipynb.fs.defs.Feature_Engineering import add_new_features
from ipynb.fs.full.Prepare_data import load_csv_into_df
from ipynb.fs.defs.Feature_Engineering import preprocess_dataset
from ipynb.fs.full.Prepare_data import prepare_and_add_labels
from ipynb.fs.full.Prepare_data import prepare_text
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# create sidebar to upload the csv file and display the possible labels
st.set_page_config(page_title="Car Part Classification", page_icon="plots_images/logos/BMWGrey.svg")

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

name, authentication_status, username = authenticator.login('Login', 'main')

 
def df_to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.close()
    processed_data = output.getvalue()
    return processed_data


if authentication_status:
    authenticator.logout('Logout', 'main')

    st.title("Car Part Classification")
    col1, col2 = st.columns(2)
    st.sidebar.write("## Upload Excel-File")
    uploaded_file = st.sidebar.file_uploader("# Short break after upload, the prediction takes a few seconds", type="xls")
    st.sidebar.image("plots_images/logos/BMWGrey.svg", use_column_width=True)


    # CSS, to display the text centered
    st.sidebar.markdown(
        """
        <style>
        .centered-bold {
            text-align: center;
            font-weight: bold;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    dataframes = []
    # Display the uploaded file as a pandas dataframe
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, header=None, skiprows=1)
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        dataframes.append(df)
        df, ncars = prepare_and_add_labels(dataframes, save_as_excel=False)
            
        # Load model
        model_path = "models/lgbm_16052023_1729/model_9552238805970149.pkl"
        with open(model_path, "rb") as fid:
            lgbm = pickle.load(fid)

        # Load the vectorizer from the file
        vectorizer_path = "models/lgbm_16052023_1729/vectorizer.pkl"
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

        # Get the vocabulary of the training data
        vocab_path = 'models/lgbm_16052023_1729/vocabulary.pkl'
        with open(vocab_path, 'rb') as f:
            vocabulary = pickle.load(f)

        for i in range(len(df)):

            df_preprocessed, df_for_plot = preprocess_dataset(df[i], cut_percent_of_front=0.25)

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
        
        st.write(f"## Relevant car parts for the {ncars[0]}:")
        st.write(df_preprocessed)

        df_xlsx = df_to_excel(df_preprocessed)
        st.download_button(label='📥 Download Prediction',
                                        data=df_xlsx ,
                                        file_name= f'{ncars[0]}_relevant_car_parts.xlsx')
    else:
        st.write("## No file uploaded yet.")
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')