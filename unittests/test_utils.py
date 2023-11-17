import numpy as np
import pandas as pd
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

'''
def test_add_labels():
    # Define a mock dataframe for testing
    data = {'feature1': [1, 2, 3, 4, 5], 'feature2': [0.1, 0.2, 0.3, 0.4, 0.5], 'feature3': ['string1', 'string2', 'string3', 'string4', 'string5']}
    data_target = {'feature1': [1, 2, 3, 4, 5], 'feature2': [0.1, 0.2, 0.3, 0.4, 0.5], 'feature3': ['string1', 'string2', 'string3', 'string4', 'string5'], "Relevant fuer Messung": ['Nein', 'Nein', 'Nein', 'Nein', 'Nein'], "Einheitsname": ['Dummy', 'Dummy', 'Dummy', 'Dummy', 'Dummy']}
    mock_dataframe = pd.DataFrame(data)
    expected_outcome = pd.DataFrame(data_target)

    # Test the function against the expected outcome
    assert expected_outcome.equals(Preperator.add_labels(mock_dataframe))
'''
