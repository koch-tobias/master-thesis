import numpy as np
import pandas as pd
import pytest
import os
import src.data_preparation_pipeline.data_preparation as prep

def test_check_if_columns_available():

    #TEST IF THE FUNCTION RETURNS A LIST
    assert isinstance(prep.check_if_columns_available(pd.DataFrame(), []), list)

    #TEST IF THE FUNCTION WORKS CORRECTLY FOR AN EMPTY DATAFRAME AND AN EMPTY LIST OF RELEVANT FEATURES
    assert prep.check_if_columns_available(pd.DataFrame(), []) == []

    #TEST IF THE FUNCTION WORKS CORRECTLY FOR AN EMPTY DATAFRAME AND A NON-EMPTY LIST OF RELEVANT FEATURES
    assert prep.check_if_columns_available(pd.DataFrame(), ["feature1", "feature2", "feature3"]) == ["feature1", "feature2", "feature3"]

    #TEST IF THE FUNCTION WORKS CORRECTLY FOR A NON-EMPTY DATAFRAME AND A NON-EMPTY LIST OF RELEVANT FEATURES
    df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]}) 
    assert prep.check_if_columns_available(df, ["feature1", "feature2", "feature3"]) == ["feature3"]

    #TEST IF THE FUNCTION WORKS CORRECTLY FOR A NON-EMPTY DATAFRAME AND AN EMPTY LIST OF RELEVANT FEATURES
    df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]}) 
    assert prep.check_if_columns_available(df, []) == []

def test_add_labels():
    # Define a mock dataframe for testing
    data = {'feature1': [1, 2, 3, 4, 5], 'feature2': [0.1, 0.2, 0.3, 0.4, 0.5], 'feature3': ['string1', 'string2', 'string3', 'string4', 'string5']}
    data_target = {'feature1': [1, 2, 3, 4, 5], 'feature2': [0.1, 0.2, 0.3, 0.4, 0.5], 'feature3': ['string1', 'string2', 'string3', 'string4', 'string5'], "Relevant fuer Messung": ['Nein', 'Nein', 'Nein', 'Nein', 'Nein'], "Einheitsname": ['Dummy', 'Dummy', 'Dummy', 'Dummy', 'Dummy']}
    mock_dataframe = pd.DataFrame(data)
    expected_outcome = pd.DataFrame(data_target)

    # Test the function against the expected outcome
    assert expected_outcome.equals(prep.add_labels(mock_dataframe))

def test_car_part_selection():
    # Read the mock CSV file as a pandas dataframe
    mock_dataframe = pd.read_excel('unittests/mock_labeled_data.xls')
    df_new = prep.car_part_selection(mock_dataframe)

    # Define the expected outcome based on the mock dataframe and the keep_modules config
    expected_outcome_shape = (6,30)

    # Test the function against the expected outcome
    assert expected_outcome_shape == df_new.shape

def test_feature_selection():
    df_test = pd.read_excel('unittests/mock_labeled_data.xls')
    df_test_prepared = prep.car_part_selection(df_test)

    #EXPECTED RESULT
    df_expected = (14,28)
    df_result = prep.feature_selection(df_test)

    assert df_result.shape == df_expected
