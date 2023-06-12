import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
from Feature_Engineering import preprocess_dataset
from Prepare_data import prepare_and_add_labels
from config import general_params, train_settings, website_setting
import streamlit_authenticator as stauth
from sklearn import preprocessing
import os
import yaml
from yaml.loader import SafeLoader

# create sidebar to upload the csv file and display the possible labels
st.set_page_config(page_title="Car Part Identification", page_icon="plots_images/logos/Download.png")

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

@st.cache_data
def logout():
    authenticator.logout('Logout', 'main')


def get_model(folder_path):
    # Load model for relevance
    for file in os.listdir(folder_path):
        if file.startswith("model"):
            model_path =  os.path.join(folder_path, file)

    with open(model_path, "rb") as fid:
        lgbm = pickle.load(fid)

    # Load the vectorizer from the file
    vectorizer_path = folder_path + "/vectorizer.pkl"
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    # Get the vocabulary of the training data
    vocab_path = folder_path + "/vocabulary.pkl"
    with open(vocab_path, 'rb') as f:
        vocabulary = pickle.load(f) 

    return lgbm, vectorizer, vocabulary

def get_X(vocab, vectorizer):
    # Convert the vocabulary list to a dictionary
    vocabulary_dict = {word: index for index, word in enumerate(vocab)}

    # Set the vocabulary of the vectorizer to the loaded vocabulary
    vectorizer.vocabulary_ = vocabulary_dict
    X = vectorizer.transform(df_preprocessed['Benennung (bereinigt)']).toarray()

    # Combine text features with other features
    if train_settings["use_only_text"] == False:
        X = np.concatenate((X, df_preprocessed[general_params["features_for_model"]].values), axis=1)
    
    return X

def hide_streamlit_header_footer():
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    

if authentication_status:
    hide_streamlit_header_footer()

    st.title("Car Part Identification")
    col1, col2 = st.columns(2)
    uploaded_file = st.sidebar.file_uploader("Upload your Excel file here...", type="xls")
    st.sidebar.write("After a successful upload, it takes a few seconds for the AI ​​to identify the relevant car parts")

    st.sidebar.button('Logout', on_click=logout)

    st.sidebar.image("plots_images/logos/BMWGrey.svg")

    dataframes = []
    # Display the uploaded file as a pandas dataframe
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, header=None, skiprows=1)
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        dataframes.append(df)
        df, ncars = prepare_and_add_labels(dataframes)
            
        lgbm_binary, vectorizer_binary, vocabulary_binary = get_model(website_setting["model_binary"])
        lgbm_multiclass, vectorizer_multiclass, vocabulary_multiclass = get_model(website_setting["model_multiclass"])

        for i in range(len(df)):

            df_preprocessed, df_for_plot = preprocess_dataset(df[i], cut_percent_of_front=0.20)

            X_binary = get_X(vocabulary_binary, vectorizer_binary)
            probs_binary = lgbm_binary.predict_proba(X_binary)
            y_pred_binary = (probs_binary[:, 1] > 0.7).astype(int)
            #y_pred_binary = np.round(probs_binary[:, 1])

            X_multiclass = get_X(vocabulary_multiclass, vectorizer_multiclass)
            probs_multiclass = lgbm_multiclass.predict_proba(X_multiclass)
            y_pred_multiclass = probs_multiclass.argmax(axis=1)

            # Load the LabelEncoder
            with open(website_setting["model_multiclass"] + '/label_encoder.pkl', 'rb') as f:
                le = pickle.load(f) 

            y_pred_multiclass_names = le.inverse_transform(y_pred_multiclass) 

            df_preprocessed = df_preprocessed.reset_index(drop=True)
            for index, row in df_preprocessed.iterrows():
                if y_pred_binary[index] == 1: 
                    df_preprocessed.loc[index,'Relevant fuer Messung'] = 'Ja'
                else:
                    df_preprocessed.loc[index,'Relevant fuer Messung'] = 'Nein'

                df_preprocessed.loc[index,'Einheitsname'] = y_pred_multiclass_names[index]
                df_preprocessed.loc[index,'Wahrscheinlichkeit Relevanz'] = probs_binary[:, 1][index]
                df_preprocessed.loc[index,'Wahrscheinlichkeit Einheitsname'] = probs_multiclass[index, y_pred_multiclass[index]]

            df_preprocessed = df_preprocessed[df_preprocessed['Relevant fuer Messung'] == 'Ja']

            df_preprocessed = df_preprocessed.loc[:,["Sachnummer", "Benennung (dt)", "Einheitsname", "L/R-Kz.", "Wahrscheinlichkeit Relevanz", "Wahrscheinlichkeit Einheitsname"]]
        
        st.write(f"## Relevant car parts for the {ncars[0]}:")
        st.write(df_preprocessed)

        df_xlsx = df_to_excel(df_preprocessed)
        st.download_button(label='🚘 Download List',
                                        data=df_xlsx ,
                                        file_name= f'{ncars[0]}_relevant_car_parts.xlsx')
        
        st.subheader("Feedback Email Template")

        st.markdown("""
        The AI is still in the development process so it is important to detect and analyze errors. 
        Please let me know if you are missing something in the list of relevant car parts. Positive feedback also helps to improve the model. 
        \n Thank you very much for your support! 
        """)
        with st.expander("Template", expanded=False):
            st.markdown("""
            **To:** tobias.ko.koch@bmw.de

            **Subject:** Feedback AI Model (Website)

            **Body:**

            Internal Model Name: [G65]

            Model Name: [X5]

            Part Number (Sachnummer): [P0HL8W7]

            Negative Feedback: [Leider fehlt der Himmel]

            Positive Feedback: [Alle anderen Bauteile wurden richtig ausgegeben]
            """)
        
    else:
        st.subheader("Instructions for downloading the car part structure tree:")
        st.image("plots_images/Anleitung_ExcelDownload.PNG")
        st.image("plots_images/Anleitung_ExcelDownload2.PNG")
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')