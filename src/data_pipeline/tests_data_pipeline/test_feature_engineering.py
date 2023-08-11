import numpy as np
import pandas as pd
import pytest
import src.data_pipeline.feature_engineering as fe

def test_transform_boundingbox():
    # Define input parameters
    x_min = 0
    x_max = 1
    y_min = 0
    y_max = 1
    z_min = 0
    z_max = 1
    ox = 1
    oy = 2
    oz = 3
    xx = 1
    xy = 0
    xz = 0
    yx = 0
    yy = 1
    yz = 0
    zx = 0
    zy = 0
    zz = 1

    # Call the function
    result = fe.transform_boundingbox(x_min, x_max, y_min, y_max, z_min, z_max, ox, oy, oz, xx, xy, xz, yx, yy, yz, zx, zy, zz)

    # Define the expected output
    expected_result = np.array([[1, 2, 3],
                                [1, 2, 4],
                                [1, 3, 3],
                                [1, 3, 4],
                                [2, 2, 3],
                                [2, 2, 4],
                                [2, 3, 3],
                                [2, 3, 4]])

    # Compare the result with the expected output
    assert np.array_equal(result, expected_result)

def test_get_minmax():
    # Define input list of transformed bounding boxes
    list_transformed_bboxes = [
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
    ]

    # Call the function
    result = fe.get_minmax(list_transformed_bboxes)

    # Define the expected output
    expected_result = (1, 8, 2, 9, 3, 10)

    # Compare the result with the expected output
    assert result == expected_result

def test_find_valid_space():
    # Create a sample DataFrame
    df = pd.DataFrame({
        'X-Min_transf': [1, 2, 3],
        'X-Max_transf': [4, 5, 6],
        'Y-Min_transf': [7, 8, 9],
        'Y-Max_transf': [10, 11, 12],
        'Z-Min_transf': [13, 14, 15],
        'Z-Max_transf': [16, 17, 18]
    })

    # Call the function
    corners, length, width, height = fe.find_valid_space(df)

    # Check the output
    assert np.array_equal(corners, np.array([[0.5, 6.5, 12.5], [0.5, 6.5, 18.5], [0.5, 12.5, 12.5], [0.5, 12.5, 18.5], [6.5, 6.5, 12.5], [6.5, 6.5, 18.5], [6.5, 12.5, 12.5], [6.5, 12.5, 18.5]]))
    assert length == 6
    assert width == 6
    assert height == 6

    # Test with a DataFrame with only one row
    single_row_df = pd.DataFrame({
        'X-Min_transf': [1],
        'X-Max_transf': [2],
        'Y-Min_transf': [3],
        'Y-Max_transf': [4],
        'Z-Min_transf': [5],
        'Z-Max_transf': [6]
    })
    corners, length, width, height = fe.find_valid_space(single_row_df)
    assert np.array_equal(corners, np.array([[0.9, 2.9, 4.9], [0.9, 2.9, 6.1], [0.9, 4.1, 4.9], [0.9, 4.1, 6.1], [2.1, 2.9, 4.9], [2.1, 2.9, 6.1], [2.1, 4.1, 4.9], [2.1, 4.1, 6.1]]))
    assert round(length, 2) == 1.2
    assert round(width, 2) == 1.2
    assert round(height, 2) == 1.2

def test_random_centerpoint_in_valid_space():
    corners = np.array([[0.5, 6.5, 12.5], [0.5, 6.5, 18.5], [0.5, 12.5, 12.5], [0.5, 12.5, 18.5], [6.5, 6.5, 12.5], [6.5, 6.5, 18.5], [6.5, 12.5, 12.5], [6.5, 12.5, 18.5]])
    length = 1
    width = 1
    height = 1

    for _ in range(100):
        centerpoint = fe.random_centerpoint_in_valid_space(corners, length, width, height)
        assert isinstance(centerpoint, np.ndarray)
        assert centerpoint.shape == (3,)

        x_min, y_min, z_min = corners[0]
        x_max, y_max, z_max = corners[7]
        x_min = x_min + (length/2)
        x_max = x_max - (length/2)
        y_min = y_min + (width/2)
        y_max = y_max - (width/2)
        z_min = z_min + (height/2)
        z_max = z_max - (height/2)

        x, y, z = centerpoint
        assert x >= x_min
        assert x <= x_max
        assert y >= y_min
        assert y <= y_max
        assert z >= z_min
        assert z <= z_max

import pytest

def calculate_center_point(transf_bbox):
    sum_X = 0
    sum_Y = 0
    sum_Z = 0
    num_corners = len(transf_bbox)
    for xyz in transf_bbox:
        sum_X = sum_X + xyz[0]
        sum_Y = sum_Y + xyz[1]
        sum_Z = sum_Z + xyz[2]
    
    center_x = sum_X/num_corners
    center_y = sum_Y/num_corners
    center_z = sum_Z/num_corners

    return center_x, center_y, center_z

def test_calculate_center_point():
    # Test case 1: Bounding box with eight corners
    transf_bbox = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15), (16, 17, 18), (19, 20, 21), (22, 23, 24)]
    expected_center = (11.5, 12.5, 13.5)
    assert calculate_center_point(transf_bbox) == expected_center

    # Test case 2: Bounding box with four corners
    transf_bbox = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]
    expected_center = (5.5, 6.5, 7.5)
    assert calculate_center_point(transf_bbox) == expected_center

    # Test case 3: Bounding box with no corners
    transf_bbox = []
    with pytest.raises(ZeroDivisionError):
        calculate_center_point(transf_bbox)

    # Test case 4: Bounding box with one corner
    transf_bbox = [(1, 2, 3)]
    expected_center = (1, 2, 3)
    assert calculate_center_point(transf_bbox) == expected_center

    # Test case 5: Bounding box with negative coordinates
    transf_bbox = [(-1, -2, -3), (-4, -5, -6), (-7, -8, -9), (-10, -11, -12)]
    expected_center = (-5.5, -6.5, -7.5)
    assert calculate_center_point(transf_bbox) == expected_center

    # Test case 6: Bounding box with non-integer coordinates
    transf_bbox = [(1.5, 2.5, 3.5), (4.5, 2.5, 6.5), (1.5, 8.5, 9.5), (4.5, 8.5, 12.5)]
    expected_center = (3.0, 5.5, 8.0)
    assert calculate_center_point(transf_bbox) == expected_center

    # Test case 7: Bounding box with repeated corners
    transf_bbox = [(1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2, 3)]
    expected_center = (1, 2, 3)
    assert calculate_center_point(transf_bbox) == expected_center

    # Test case 8: Bounding box with non-numeric coordinates
    transf_bbox = [('a', 'b', 'c'), ('d', 'e', 'f'), ('g', 'h', 'i'), ('j', 'k', 'l')]
    with pytest.raises(TypeError):
        calculate_center_point(transf_bbox)

    # Test case 9: Bounding box with missing coordinates
    transf_bbox = [(1, 2), (3, 4), (5, 6), (7, 8)]
    with pytest.raises(IndexError):
        calculate_center_point(transf_bbox)

def test_calculate_lwh():
    transformed_boundingbox = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    length, width, height = fe.calculate_lwh(transformed_boundingbox)
    assert length == 6
    assert width == 6
    assert height == 6

    transformed_boundingbox = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    length, width, height = fe.calculate_lwh(transformed_boundingbox)
    assert length == 0
    assert width == 0
    assert height == 0

    transformed_boundingbox = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]
    length, width, height = fe.calculate_lwh(transformed_boundingbox)
    assert length == 9
    assert width == 9
    assert height == 9

def test_rotation_to_orientation():
    rotation_matrix = np.array([[0.5, -0.1464, 0.8536], [0.5, 0.8536, -0.1464], [-0.7071, 0.5, 0.5]])
    expected_output = [45.0, 45.0, 45.0]

    theta_x, theta_y, theta_z = fe.rotation_to_orientation(rot_mat=rotation_matrix)   
    degrees = [round(np.degrees(theta_x),1), round(np.degrees(theta_y),1), round(np.degrees(theta_z),1)]

    assert degrees == expected_output

def test_prepare_text(): 
    input_text = "ZB-34232/RE LI9090/TAB HIMMEL SHD." 
    expected_output = "HIMMEL SHD" 
    assert fe.prepare_text(input_text) == expected_output

    input_text = "This is a test A.F.-51232/AF designation." 
    expected_output = "THIS IS TEST DESIGNATION" 
    assert fe.prepare_text(input_text) == expected_output


def test_clean_text():
   # Create a test dataframe
   df = pd.DataFrame({
       'Benennung (dt)': ['This is a Test text! 5', 'ZB LL RE AF 123 MD']
   })
   
   # Apply the clean_text function
   df_cleaned = fe.clean_text(df)
   
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
   X = fe.nchar_text_to_vec(sample_data, model_folder)
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
   vocabulary = fe.get_vocabulary(sample_column)
   # Check that the output is a list of strings
   assert isinstance(vocabulary, list)
   assert all(isinstance(word, str) for word in vocabulary)
   # Check that the vocabulary has the expected length and content
   assert len(vocabulary) == len(expected_vocabulary)
   assert set(vocabulary) == set(expected_vocabulary)
   # Check that the function is case-insensitive
   assert 'dachantenne' not in vocabulary