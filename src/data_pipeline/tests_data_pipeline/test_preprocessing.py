import numpy as np
import pandas as pd
import pytest
import os
import src.data_pipeline.preprocessing as pp

def test_check_nan_values():

    df = pd.DataFrame({'col1': [1, 2, 3, 4, 5], 'col2': [6, 7, 8, None, 10], 'col3': [11, None, 13, 14, 15]})
    assert pp.check_nan_values(df) == ['col2', 'col3']

    df2 = pd.DataFrame({'col1': [1, 2, 3, 4, 5], 'col2': [6, 7, 8, 9, 10], 'col3': [11, 12, 13, 14, 15]})
    assert pp.check_nan_values(df2) == []

    df3 = pd.DataFrame({'col1': [None, None, None, None, None], 'col2': [None, None, None, None, None], 'col3': [None, None, None, None, None]})
    assert pp.check_nan_values(df3) == ['col1', 'col2', 'col3']

def test_combine_dataframes():

    df1 = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}) 
    df2 = pd.DataFrame({'col1': [7, 8, 9], 'col2': [10, 11, 12]}) 
    df3 = pd.DataFrame({'col1': [13, 14, 15], 'col2': [16, 17, 18]})
    assert pp.combine_dataframes([df1, df2, df3]).equals(pd.DataFrame({'col1': [1, 2, 3, 7, 8, 9, 13, 14, 15], 'col2': [4, 5, 6, 10, 11, 12, 16, 17, 18]}))

    df4 = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}) 
    df5 = pd.DataFrame({'col3': [7, 8, 9], 'col4': [10, 11, 12]})
    with pytest.raises(ValueError): pp.combine_dataframes([df4, df5])

    df6 = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, None, 6]})
    assert pp.combine_dataframes([df1, df2, df6]).equals(pd.DataFrame({'col1': [1, 2, 3, 7, 8, 9, 1, 2, 3], 'col2': [4, 5, 6, 10, 11, 12, 4, None, 6]}))


def test_check_if_columns_available():

    relevant_features = ['col1', 'col2', 'col3']
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9]})
    assert pp.check_if_columns_available(df, relevant_features) == []

    df2 = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    assert pp.check_if_columns_available(df2, relevant_features) == ['col3']

    df3 = pd.DataFrame({'col4': [1, 2, 3], 'col5': [4, 5, 6]})
    assert pp.check_if_columns_available(df3, relevant_features) == ['col1', 'col2', 'col3']

    df4 = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9], 'col4': [10, 11, 12]})
    assert pp.check_if_columns_available(df4, relevant_features) == []

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

