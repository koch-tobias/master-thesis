import numpy as np
import pandas as pd

import pytest
from src.data_preprocessing_pipeline.feature_engineering import Feature_Engineering as fe

def test_transform_boundingbox(): 
    x_min, x_max, y_min, y_max, z_min, z_max = 0, 1, 0, 1, 0, 1 
    ox, oy, oz = 1, 1, 1
    xx, xy, xz = 1, 0, 0
    yx, yy, yz = 0, 1, 0
    zx, zy, zz = 0, 0, 1 
    transformed_corners, rotation_matrix = fe.transform_boundingbox(x_min, x_max, y_min, y_max, z_min, z_max, ox, oy, oz, xx, xy, xz, yx, yy, yz, zx, zy, zz)

    expected_corners = np.array([[1., 1., 1.], [1., 1., 2.], [1., 2., 1.], [1., 2., 2.], [2., 1., 1.], [2., 1., 2.], [2., 2., 1.], [2., 2., 2.]])
    expected_rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    assert np.array_equal(transformed_corners, expected_corners)
    assert np.array_equal(rotation_matrix, expected_rotation_matrix)

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

def test_calculate_transformed_corners(): 
    center_point = [1, 1, 1] 
    length, width, height = 1, 1, 1 
    theta_x, theta_y, theta_z = 0, 0, 0

    x_min, x_max, y_min, y_max, z_min, z_max = fe.calculate_transformed_corners(center_point, length, width, height, theta_x, theta_y, theta_z)

    expected_corners = np.array([0.5, 1.5, 0.5, 1.5, 0.5, 1.5])

    assert np.array_equal([x_min, x_max, y_min, y_max, z_min, z_max], expected_corners)


def test_calculate_center_point():
    # Test case 1: Bounding box with eight corners
    transf_bbox = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15), (16, 17, 18), (19, 20, 21), (22, 23, 24)]
    expected_center = (11.5, 12.5, 13.5)
    assert fe.calculate_center_point(transf_bbox) == expected_center

    # Test case 2: Bounding box with four corners
    transf_bbox = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]
    expected_center = (5.5, 6.5, 7.5)
    assert fe.calculate_center_point(transf_bbox) == expected_center

    # Test case 3: Bounding box with no corners
    transf_bbox = []
    with pytest.raises(ZeroDivisionError):
        fe.calculate_center_point(transf_bbox)

    # Test case 4: Bounding box with one corner
    transf_bbox = [(1, 2, 3)]
    expected_center = (1, 2, 3)
    assert fe.calculate_center_point(transf_bbox) == expected_center

    # Test case 5: Bounding box with negative coordinates
    transf_bbox = [(-1, -2, -3), (-4, -5, -6), (-7, -8, -9), (-10, -11, -12)]
    expected_center = (-5.5, -6.5, -7.5)
    assert fe.calculate_center_point(transf_bbox) == expected_center

    # Test case 6: Bounding box with non-integer coordinates
    transf_bbox = [(1.5, 2.5, 3.5), (4.5, 2.5, 6.5), (1.5, 8.5, 9.5), (4.5, 8.5, 12.5)]
    expected_center = (3.0, 5.5, 8.0)
    assert fe.calculate_center_point(transf_bbox) == expected_center

    # Test case 7: Bounding box with repeated corners
    transf_bbox = [(1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2, 3)]
    expected_center = (1, 2, 3)
    assert fe.calculate_center_point(transf_bbox) == expected_center

    # Test case 8: Bounding box with non-numeric coordinates
    transf_bbox = [('a', 'b', 'c'), ('d', 'e', 'f'), ('g', 'h', 'i'), ('j', 'k', 'l')]
    with pytest.raises(TypeError):
        fe.calculate_center_point(transf_bbox)

    # Test case 9: Bounding box with missing coordinates
    transf_bbox = [(1, 2), (3, 4), (5, 6), (7, 8)]
    with pytest.raises(IndexError):
        fe.calculate_center_point(transf_bbox)

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
