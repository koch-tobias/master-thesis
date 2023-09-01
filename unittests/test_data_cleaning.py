import numpy as np
import pandas as pd
import pytest
from src.data_preprocessing_pipeline.data_cleaning import DataCleaner

def test_outlier_detection():
    #TEST IF THE FUNCTION WORKS CORRECTLY FOR A DATAFRAME WITH MULTIPLE ROWS AND MULTIPLE RELEVANT FEATURES
    df = pd.DataFrame({"X-Max_transf": [1, 2, 3, 4, 100], "X-Min_transf": [2, 3, 4, 5, -100]}) 
    assert df.shape[0] == DataCleaner.outlier_detection(df).shape[0]
    assert DataCleaner.outlier_detection(df).iloc[4]["X-Max"] == 0 

    #TEST IF THE FUNCTION WORKS CORRECTLY FOR A DATAFRAME WITH ONE ROW
    df = pd.DataFrame({"X-Max_transf": [1], "X-Min_transf": [2]}) 
    assert df.shape[0] == DataCleaner.outlier_detection(df).shape[0]

    #TEST IF THE FUNCTION RETURNS A PANDAS DATAFRAME
    assert isinstance(DataCleaner.outlier_detection(pd.DataFrame()), pd.DataFrame)

    #TEST IF THE FUNCTION WORKS CORRECTLY FOR AN EMPTY DATAFRAME
    df = pd.DataFrame() 
    assert DataCleaner.outlier_detection(df).empty

def test_prepare_text(): 
    input_text = "ZB-34232/RE LI9090/TAB HIMMEL SHD." 
    expected_output = "HIMMEL SHD" 
    assert DataCleaner.prepare_text(input_text) == expected_output

    input_text = "This is a test A.F.-51232/AF designation." 
    expected_output = "THIS IS TEST DESIGNATION" 
    assert DataCleaner.prepare_text(input_text) == expected_output

def test_clean_text():
   # Create a test dataframe
   df = pd.DataFrame({
                        'Benennung (dt)': ['This is a Test text! 5', 'ZB LL RE AF 123 MD']
                    })
   
   # Apply the clean_text function
   df_cleaned = DataCleaner.clean_text(df)
   
   # Test the transformed text of the first row
   assert df_cleaned['Benennung (bereinigt)'][0] == 'THIS IS TEST TEXT'

   # Test the transformed text of the second row
   assert df_cleaned['Benennung (bereinigt)'][1] == ''


def test_nchar_text_to_vec():
   sample_data = pd.DataFrame({
                                'Benennung (bereinigt)': [
                                    'DACHANTENNE', 'HECKKLAPPE AUSSENHAUT', 'FRONTSCHEIBE', 'HIMMEL ND'
                                ],
                                "Relevant fuer Messung":["Ja", "Ja", "Ja", "Ja"]
                            })
   model_folder = ""
   X = DataCleaner.nchar_text_to_vec(sample_data, model_folder)
   # Check that the output is a numpy array
   assert isinstance(X, np.ndarray)
   # Check that the array has the expected shape
   assert X.shape == (4, 210)
   # Check that the array is not all zeros
   assert np.any(X)

@pytest.fixture
def sample_column():
   return pd.Series(['DACHANTENNE', 'HECKKLAPPE AUSSENHAUT', 'FRONTSCHEIBE', 'HIMMEL ND'])

def test_get_vocabulary(sample_column):
   expected_vocabulary = ['DACHANTENNE', 'HECKKLAPPE', 'AUSSENHAUT', 'FRONTSCHEIBE', 'HIMMEL', 'ND']
   vocabulary = DataCleaner.get_vocabulary(sample_column)
   # Check that the output is a list of strings
   assert isinstance(vocabulary, list)
   assert all(isinstance(word, str) for word in vocabulary)
   # Check that the vocabulary has the expected length and content
   assert len(vocabulary) == len(expected_vocabulary)
   assert set(vocabulary) == set(expected_vocabulary)
   # Check that the function is case-insensitive
   assert 'dachantenne' not in vocabulary
