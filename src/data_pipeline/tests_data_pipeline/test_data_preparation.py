import numpy as np
import pandas as pd
import pytest
import os
import src.data_pipeline.data_preparation as prep

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