import streamlit as st
import pandas as pd
import numpy as np
import os 
from datetime import datetime
from shutil import copy2

# initialize labels
labels = {0: "Label 1", 1: "Label 2", 2: "Label 3", 3: "Label 4", 4: "Label 5"}
labels_keys = labels.keys()

# create sidebar to upload the csv file and display the possible labels
st.title("Data Labeling Tool")
col1, col2 = st.columns(2)
st.sidebar.write("## Excel-Datei hochladen")
uploaded_file = st.sidebar.file_uploader("# Excel-Datei hochladen", type="xlsx")

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

st.sidebar.write("## Mögliche Labels:")
col1, col2 = st.sidebar.columns((1, 1))
for i in range(len(labels)):
    with col1:
        st.write(f"<div class='centered-bold'>{i}:</div>", unsafe_allow_html=True)
    with col2:
        st.write(f"<div class='centered-bold'>{labels[i]}</div>", unsafe_allow_html=True)
