import pandas as pd
import src.data_pipeline.augmentation as aug

def test_random_order():
    designation = "MD HIMMEL SKYROOF US A.F."
    words = designation.split()

    new_designation = aug.random_order(designation)

    if words in new_designation.split():
        assert new_designation != designation
        assert len(new_designation) == len(designation)

def test_remove_prefix():
    response1 = "Answer: HIMMEL SKYROOF US VERSION"
    response2 = "HIMMEL SKYROOF US VERSION"

    target = "HIMMEL SKYROOF US VERSION"

    new_response1 = aug.remove_prefix(response1)
    new_response2 = aug.remove_prefix(response2)

    assert target == new_response1
    assert target == new_response2

def test_augmented_boundingbox():
    df_original = pd.read_csv("src/data_pipeline/tests_data_pipeline/augmentation/df_orignal.csv", index_col=False)
    df_temp = pd.read_csv("src/data_pipeline/tests_data_pipeline/augmentation/df_temp.csv", index_col=False)
    target_result = pd.read_csv("src/data_pipeline/tests_data_pipeline/augmentation/result.csv", index_col=False)

    result = aug.augmented_boundingbox(df_original, df_temp)

    assert result.shape == target_result.shape

def test_data_augmentation():
    df_original = pd.read_csv("src/data_pipeline/tests_data_pipeline/augmentation/df_orignal.csv", index_col=False)
    target_result = pd.read_csv("src/data_pipeline/tests_data_pipeline/augmentation/target_data_augmentation.csv", index_col=False)

    result = aug.data_augmentation(df_original)

    assert result.shape == target_result.shape
