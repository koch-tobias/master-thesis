import pandas as pd
import numpy as np
from src.data_preprocessing_pipeline.data_preprocessing import DataGenerator

def test_get_weight_factor():
    # Test case 1: binary_model = False
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", "Ja", "Ja", "Ja", "Ja", "Ja"]
    })
    binary_model = False
    expected_result = {1: 2, 2: 2, 3: 1}
    assert DataGenerator.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 2: binary_model = True
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", "Ja", "Nein", "Nein", "Nein", "Nein"]
    })
    binary_model = True
    expected_result = 1
    assert DataGenerator.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 3: binary_model = True, no "Ja" values in df
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Nein", "Nein", "Nein", "Nein", "Nein", "Nein", "Nein"]
    })
    binary_model = True
    expected_result = 0
    assert DataGenerator.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 4: binary_model = False, empty y and df
    y = np.array([])
    df = pd.DataFrame(columns=["Einheitsname", "Relevant fuer Messung"])
    binary_model = False
    expected_result = {}
    assert DataGenerator.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 5: binary_model = True, empty df
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame(columns=["Einheitsname", "Relevant fuer Messung"])
    binary_model = True
    expected_result = 0
    assert DataGenerator.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 6: binary_model = False, all "Ja" values in df
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", "Ja", "Ja", "Ja", "Ja", "Ja"]
    })
    binary_model = False
    expected_result = {1: 2, 2: 2, 3: 1}
    assert DataGenerator.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 7: binary_model = True, all "Nein" values in df
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Nein", "Nein", "Nein", "Nein", "Nein", "Nein", "Nein"]
    })
    binary_model = True
    expected_result = 0
    assert DataGenerator.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 8: binary_model = False, y contains only one unique value
    y = np.array([1, 1, 1, 1, 1])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", "Ja", "Ja", "Ja"]
    })
    binary_model = False
    expected_result = {1: 1}
    assert DataGenerator.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 10: binary_model = False, y and df contain only one unique value
    y = np.array([1, 1, 1, 1, 1])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", "Ja", "Ja", "Ja"]
    })
    binary_model = False
    expected_result = {1: 1}
    assert DataGenerator.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 11: binary_model = True, df is empty
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame(columns=["Einheitsname", "Relevant fuer Messung"])
    binary_model = True
    expected_result = 0
    assert DataGenerator.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 12: binary_model = False, y is empty
    y = np.array([])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", "Ja", "Ja", "Ja", "Ja", "Ja"]
    })
    binary_model = False
    expected_result = {}
    assert DataGenerator.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 13: binary_model = True, df contains NaN values
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", np.nan, "Nein", "Nein", "Nein", "Nein"]
    })
    binary_model = True
    expected_result = 2
    assert DataGenerator.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 14: binary_model = False, df contains NaN values
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", np.nan, "Ja", "Ja", "Ja", "Ja"]
    })
    binary_model = False
    expected_result = {1: 2, 2: 2, 3: 1}
    assert DataGenerator.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 15: binary_model = True, df contains non-numeric values
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", "Ja", "Nein", "Nein", "Nein", "Invalid"]
    })
    binary_model = True
    expected_result = 1
    assert DataGenerator.get_weight_factor(y, df, binary_model) == expected_result

    # Test case 16: binary_model = False, df contains non-numeric values
    y = np.array([1, 2, 1, 2, 3, 3, 3])
    df = pd.DataFrame({
        "Relevant fuer Messung": ["Ja", "Ja", "Ja", "Ja", "Ja", "Ja", "Invalid"]
    })
    binary_model = False
    expected_result = {1: 2, 2: 2, 3: 1}
    assert DataGenerator.get_weight_factor(y, df, binary_model) == expected_result