import numpy as np
import pandas as pd
import pytest
import os
import src.utils as utils

def test_check_nan_values():
    #TEST CASE: WHEN THERE ARE NAN VALUES IN THE INPUT DATAFRAME
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}) 
    relevant_features = ['A', 'B', 'C'] 
    ncar = 'car_1'
    assert utils.check_nan_values(df, relevant_features, ncar) == []

    #TEST CASE: WHEN THERE ARE NO NAN VALUES IN THE INPUT DATAFRAME
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, np.nan, 6], 'C': [7, 8, 9]}) 
    relevant_features = ['A', 'B', 'C'] 
    ncar = 'car_2'
    expected_result = ['B'] 
    assert utils.check_nan_values(df, relevant_features, ncar) == expected_result

def test_combine_dataframes():

    #TEST CASE 1: ALL DATAFRAMES HAVE SAME COLUMNS AND NO NAN VALUES
    df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}) 
    df2 = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]}) 
    relevant_features = ['A', 'B']
    dataframes = [df1, df2] 
    ncars = ['G3', 'G4']

    expected_result = pd.DataFrame({'A': [1, 2, 3, 7, 8, 9], 'B': [4, 5, 6, 10, 11, 12]}) 
    assert utils.combine_dataframes(dataframes, relevant_features=relevant_features, ncars=ncars).equals(expected_result)
