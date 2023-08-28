import numpy as np
import pandas as pd
import pytest
import src.data_preprocessing_pipeline.data_cleaning as pp

def test_outlier_detection():
    #TEST IF THE FUNCTION WORKS CORRECTLY FOR A DATAFRAME WITH MULTIPLE ROWS AND MULTIPLE RELEVANT FEATURES
    df = pd.DataFrame({"X-Max_transf": [1, 2, 3, 4, 100], "X-Min_transf": [2, 3, 4, 5, -100]}) 
    assert df.shape[0] == pp.outlier_detection(df).shape[0]
    assert pp.outlier_detection(df).iloc[4]["X-Max"] == 0 

    #TEST IF THE FUNCTION WORKS CORRECTLY FOR A DATAFRAME WITH ONE ROW
    df = pd.DataFrame({"X-Max_transf": [1], "X-Min_transf": [2]}) 
    assert df.shape[0] == pp.outlier_detection(df).shape[0]

    #TEST IF THE FUNCTION RETURNS A PANDAS DATAFRAME
    assert isinstance(pp.outlier_detection(pd.DataFrame()), pd.DataFrame)

    #TEST IF THE FUNCTION WORKS CORRECTLY FOR AN EMPTY DATAFRAME
    df = pd.DataFrame() 
    assert pp.outlier_detection(df).empty

def test_prepare_text(): 
    input_text = "ZB-34232/RE LI9090/TAB HIMMEL SHD." 
    expected_output = "HIMMEL SHD" 
    assert pp.prepare_text(input_text) == expected_output

    input_text = "This is a test A.F.-51232/AF designation." 
    expected_output = "THIS IS TEST DESIGNATION" 
    assert pp.prepare_text(input_text) == expected_output

def test_clean_text():
   # Create a test dataframe
   df = pd.DataFrame({
                        'Benennung (dt)': ['This is a Test text! 5', 'ZB LL RE AF 123 MD']
                    })
   
   # Apply the clean_text function
   df_cleaned = pp.clean_text(df)
   
   # Test the transformed text of the first row
   assert df_cleaned['Benennung (bereinigt)'][0] == 'THIS IS TEST TEXT'

   # Test the transformed text of the second row
   assert df_cleaned['Benennung (bereinigt)'][1] == ''


def test_nchar_text_to_vec():
   sample_data = pd.DataFrame({
                                'Benennung (bereinigt)': [
                                    'DACHANTENNE', 'HECKKLAPPE AUSSENHAUT', 'FRONTSCHEIBE', 'HIMMEL ND'
                                ]
                            })
   model_folder = ""
   X = pp.nchar_text_to_vec(sample_data, model_folder)
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
   vocabulary = pp.get_vocabulary(sample_column)
   # Check that the output is a list of strings
   assert isinstance(vocabulary, list)
   assert all(isinstance(word, str) for word in vocabulary)
   # Check that the vocabulary has the expected length and content
   assert len(vocabulary) == len(expected_vocabulary)
   assert set(vocabulary) == set(expected_vocabulary)
   # Check that the function is case-insensitive
   assert 'dachantenne' not in vocabulary

def test_get_weight_factor():
    # Test case 1: binary_model = False
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", "Ja", "Ja", "Ja", "Ja", "Ja"]
    })
    binary_model = False
    expected_result = {1: 2, 2: 2, 3: 1}
    assert pp.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 2: binary_model = True
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", "Ja", "Nein", "Nein", "Nein", "Nein"]
    })
    binary_model = True
    expected_result = 1
    assert pp.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 3: binary_model = True, no "Ja" values in df
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Nein", "Nein", "Nein", "Nein", "Nein", "Nein", "Nein"]
    })
    binary_model = True
    expected_result = 0
    assert pp.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 4: binary_model = False, empty y and df
    y = np.array([])
    df = pd.DataFrame(columns=["Einheitsname", "Relevant fuer Messung"])
    binary_model = False
    expected_result = {}
    assert pp.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 5: binary_model = True, empty df
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame(columns=["Einheitsname", "Relevant fuer Messung"])
    binary_model = True
    expected_result = 0
    assert pp.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 6: binary_model = False, all "Ja" values in df
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", "Ja", "Ja", "Ja", "Ja", "Ja"]
    })
    binary_model = False
    expected_result = {1: 2, 2: 2, 3: 1}
    assert pp.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 7: binary_model = True, all "Nein" values in df
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Nein", "Nein", "Nein", "Nein", "Nein", "Nein", "Nein"]
    })
    binary_model = True
    expected_result = 0
    assert pp.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 8: binary_model = False, y contains only one unique value
    y = np.array([1, 1, 1, 1, 1])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", "Ja", "Ja", "Ja"]
    })
    binary_model = False
    expected_result = {1: 1}
    assert pp.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 10: binary_model = False, y and df contain only one unique value
    y = np.array([1, 1, 1, 1, 1])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", "Ja", "Ja", "Ja"]
    })
    binary_model = False
    expected_result = {1: 1}
    assert pp.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 11: binary_model = True, df is empty
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame(columns=["Einheitsname", "Relevant fuer Messung"])
    binary_model = True
    expected_result = 0
    assert pp.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 12: binary_model = False, y is empty
    y = np.array([])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", "Ja", "Ja", "Ja", "Ja", "Ja"]
    })
    binary_model = False
    expected_result = {}
    assert pp.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 13: binary_model = True, df contains NaN values
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", np.nan, "Nein", "Nein", "Nein", "Nein"]
    })
    binary_model = True
    expected_result = 2
    assert pp.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 14: binary_model = False, df contains NaN values
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", np.nan, "Ja", "Ja", "Ja", "Ja"]
    })
    binary_model = False
    expected_result = {1: 2, 2: 2, 3: 1}
    assert pp.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 15: binary_model = True, df contains non-numeric values
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", "Ja", "Nein", "Nein", "Nein", "Invalid"]
    })
    binary_model = True
    expected_result = 1
    assert pp.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 16: binary_model = False, df contains non-numeric values
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", "Ja", "Ja", "Ja", "Ja", "Invalid"]
    })
    binary_model = False
    expected_result = {1: 2, 2: 2, 3: 1}
    assert pp.get_weight_factor(y, df, binary_model) == expected_result
