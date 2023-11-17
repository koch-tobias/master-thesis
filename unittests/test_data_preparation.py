import pandas as pd
from src.data_preparation_pipeline.data_preparation import Preperator

def test_check_if_columns_available():

    #TEST IF THE FUNCTION RETURNS A LIST
    assert isinstance(Preperator.check_if_columns_available(pd.DataFrame(), []), list)

    #TEST IF THE FUNCTION WORKS CORRECTLY FOR AN EMPTY DATAFRAME AND AN EMPTY LIST OF RELEVANT FEATURES
    assert Preperator.check_if_columns_available(pd.DataFrame(), []) == []

    #TEST IF THE FUNCTION WORKS CORRECTLY FOR AN EMPTY DATAFRAME AND A NON-EMPTY LIST OF RELEVANT FEATURES
    assert Preperator.check_if_columns_available(pd.DataFrame(), ["feature1", "feature2", "feature3"]) == ["feature1", "feature2", "feature3"]

    #TEST IF THE FUNCTION WORKS CORRECTLY FOR A NON-EMPTY DATAFRAME AND A NON-EMPTY LIST OF RELEVANT FEATURES
    df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]}) 
    assert Preperator.check_if_columns_available(df, ["feature1", "feature2", "feature3"]) == ["feature3"]

    #TEST IF THE FUNCTION WORKS CORRECTLY FOR A NON-EMPTY DATAFRAME AND AN EMPTY LIST OF RELEVANT FEATURES
    df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]}) 
    assert Preperator.check_if_columns_available(df, []) == []

def test_car_part_selection():
    # Read the mock CSV file as a pandas dataframe
    mock_dataframe = pd.read_excel('unittests/mock_labeled_data.xls')
    df_new = Preperator.car_part_selection(mock_dataframe)

    # Define the expected outcome based on the mock dataframe and the keep_modules config
    expected_outcome_shape = (10,36)

    # Test the function against the expected outcome
    assert expected_outcome_shape == df_new.shape

def test_feature_selection():
    df_test = pd.read_excel('unittests/mock_labeled_data.xls')
    df_test_prepared = Preperator.car_part_selection(df_test)

    #EXPECTED RESULT
    df_expected = (10,35)
    df_result = Preperator.feature_selection(df_test_prepared)

    assert df_result.shape == df_expected
