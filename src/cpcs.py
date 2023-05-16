import streamlit as st
import pandas as pd
import numpy as np
import os 
from datetime import datetime
from shutil import copy2

# create sidebar to upload the csv file and display the possible labels
st.set_page_config(page_title="Car Part Classification", page_icon="../plots_images/logos/BMWGrey.svg")

st.image("../plots_images/logos/BMW_Group_Grey.svg", use_column_width=False, width=5, output_format="SVG")

st.title("Car Part Classification")
col1, col2 = st.columns(2)
st.sidebar.write("## Excel-Datei hochladen")
uploaded_file = st.sidebar.file_uploader("# Excel-Datei hochladen", type="xlsx")
st.sidebar.image("../plots_images/logos/BMWGrey.svg", use_column_width=True)


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

