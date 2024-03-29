import pandas as pd

import streamlit as st
import streamlit_authenticator as stauth

import yaml
from yaml.loader import SafeLoader
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb

from inference import Identifier
from src.utils import read_file


with open('src/deployment/config_website.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# create sidebar to upload the csv file and display the possible labels
st.set_page_config(page_title="Car Part Identification", page_icon="images/logos/Download.png")

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

@st.cache_resource
def logout():
    authenticator.logout('Logout', 'main')

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

    uploaded_file = st.sidebar.file_uploader("Upload your Excel file here...", type="xls")
    st.sidebar.write("After a successful upload, it takes a few seconds for the AI ​​to identify the relevant car parts")

    st.sidebar.button('Logout', on_click=logout)

    st.sidebar.image("images/logos/BMWGrey.svg")

    dataframes = []
    # Display the uploaded file as a pandas dataframe
    if uploaded_file is not None:
        st.session_state['uploaded_file'] = uploaded_file

    run_identification = True
    if 'uploaded_file' in st.session_state:
        if run_identification:
            df, ncar = read_file(st.session_state['uploaded_file'], raw=True)
            df_preprocessed, df_relevant_parts, einheitsname_not_found, ncar = Identifier.classification_on_new_data(df)
            df_relevant_parts.rename(columns={'L/R-Kz.':'Linke/Rechte Ausfuehrung'}, inplace=True)


            if username == "admin":
                df_prediction = df_relevant_parts.loc[:,["Sachnummer", "Benennung (dt)", "Zeichnungsindex", "Doku-Teil", "Alternative", "Dok-Format", "Einheitsname", "Wahrscheinlichkeit Relevanz", "Wahrscheinlichkeit Einheitsname"]]
            else:
                df_prediction = df_relevant_parts.loc[:,["Sachnummer", "Benennung (dt)", "Zeichnungsindex", "Doku-Teil", "Alternative", "Dok-Format", "Einheitsname"]]

            st.write(f"## Relevant car parts for the {ncar}:")
            st.write(df_prediction)
            df_xlsx = df_to_excel(df_prediction)
            st.download_button(label='🚘 Download List',
                                            data=df_xlsx ,
                                            file_name= f'{ncar}_relevant_car_parts.xlsx')

            if len(einheitsname_not_found) > 0:
                st.write("The following parts are not found in the uploaded data. Please check manually: ")
            
            col1, col2, col3 = st.columns([1, 5, 1])
            with col2:
                if len(einheitsname_not_found) > 0:
                    st.write("\t" + "- " + "\n- ".join(einheitsname_not_found))

            st.subheader("Feedback Email Template")

            st.markdown("""
            The AI is still in the development process so it is important to detect and analyze errors. 
            Please let me know if you are missing something in the list of relevant car parts. Positive feedback also helps to improve the model. 
            \n Thank you very much for your support! 
            """)
            with st.expander("Template", expanded=False):
                st.markdown("""
                **To:** tobias.ko.koch@bmw.de and julian.chander@bmw.de

                **Subject:** Feedback AI Model (Website)

                **Body:**

                Internal Model Name: [G65]

                Model Name: [X5]

                Part Number (Sachnummer): [P0HL8W7]

                Negative Feedback: [Leider fehlt der Himmel]

                Positive Feedback: [Alle anderen Bauteile wurden richtig ausgegeben]
                """)

            run_identification = False
        
    else:
        st.subheader(f"Welcome to CaPI, please follow the instructions for downloading the car part structure tree:")
        st.image("images/Instruction_data_collection/Anleitung_ExcelDownload.PNG")
        st.image("images/Instruction_data_collection/Anleitung_ExcelDownload2.PNG")
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')