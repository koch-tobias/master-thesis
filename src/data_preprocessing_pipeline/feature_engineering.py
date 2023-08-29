import numpy as np
import pandas as pd
import math

import yaml
from yaml.loader import SafeLoader
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

from loguru import logger

class Feature_Engineering:
    # PyTest exist
    @staticmethod
    def transform_boundingbox(x_min, x_max, y_min, y_max, z_min, z_max, ox, oy, oz, xx, xy, xz, yx, yy, yz, zx, zy, zz) -> np.array:
        ''' 
        This function takes the minimum and maximum coordinates of a bounding box as well as rotation and translation parameters as inputs. It returns the new coordinates of the bounding box after applying the provided rotation and translation. 
        Args:
            x_min: a float which corresponds to the minimum x-coordinate value of the bounding box
            x_max: a float which corresponds to the maximum x-coordinate value of the bounding box
            y_min: a float which corresponds to the minimum y-coordinate value of the bounding box
            y_max: a float which corresponds to the maximum y-coordinate value of the bounding box
            z_min: a float which corresponds to the minimum z-coordinate value of the bounding box
            z_max: a float which corresponds to the maximum z-coordinate value of the bounding box
            ox: a float which corresponds to the x-coordinate value of the translation vector
            oy: a float which corresponds to the y-coordinate value of the translation vector
            oz: a float which corresponds to the z-coordinate value of the translation vector
            xx: a float which corresponds to the (1,1) element of the rotation matrix
            xy: a float which corresponds to the (1,2) element of the rotation matrix
            xz: a float which corresponds to the (1,3) element of the rotation matrix
            yx: a float which corresponds to the (2,1) element of the rotation matrix
            yy: a float which corresponds to the (2,2) element of the rotation matrix
            yz: a float which corresponds to the (2,3) element of the rotation matrix
            zx: a float which corresponds to the (2,3) element of the rotation matrix
            zy: a float which corresponds to the (2,3) element of the rotation matrix
            zz: a float which corresponds to the (2,3) element of the rotation matrix
        Return:
            transformed corners, rotation matrix
        '''
        # Create an array of corner points of the bounding box
        corners = np.array([[x_min, y_min, z_min],
                            [x_min, y_min, z_max],
                            [x_min, y_max, z_min],
                            [x_min, y_max, z_max],
                            [x_max, y_min, z_min],
                            [x_max, y_min, z_max],
                            [x_max, y_max, z_min],
                            [x_max, y_max, z_max]])

        # Create a rotation matrix using the provided rotation values
        rotation_matrix = np.array([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]])

        # Create a translation vector using the provided shift values
        shift_vec = np.array([ox, oy, oz])

        # Apply rotation to the corner points
        rotated_corners = np.dot(corners, rotation_matrix)

        # Apply translation to the rotated corner points
        transformed_corners = rotated_corners + shift_vec

        # Return the transformed corner points and the rotation matrix
        return transformed_corners, rotation_matrix

    # PyTest exist
    @staticmethod
    def get_minmax(list_transformed_bboxes: list) -> float:
        '''
        This function takes a list of transformed bounding boxes and returns the minimum and maximum coordinates for each axis 
        Args:
            list_transformed_bboxes: a list of transformed bounding boxes. Each list element should be a 2D numpy array of shape (8,3) where each row corresponds to a 3D coordinate in (x,y,z) order which describes the corners of a bounding box 
        Return:
            x_min: a float which corresponds to the minimum x-coordinate value
            x_max: a float which corresponds to the maximum x-coordinate value
            y_min: a float which corresponds to the minimum y-coordinate value
            y_max: a float which corresponds to the maximum y-coordinate value
            z_min: a float which corresponds to the minimum z-coordinate value
            z_max: a float which corresponds to the maximum z-coordinate value 
        '''
        # Initialize the minimum and maximum coordinates to infinity and negative infinity, respectively
        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        z_min = np.inf
        z_max = -np.inf

        # Iterate over each transformed bounding box in the list
        for arr in list_transformed_bboxes:
            # Extract the x, y, and z coordinates from the transformed bounding box
            x_coords = arr[:, 0]
            y_coords = arr[:, 1]
            z_coords = arr[:, 2]
            
            # Update the minimum and maximum coordinates for each axis
            x_min = min(x_min, np.min(x_coords))
            x_max = max(x_max, np.max(x_coords))
            y_min = min(y_min, np.min(y_coords))
            y_max = max(y_max, np.max(y_coords))
            z_min = min(z_min, np.min(z_coords))
            z_max = max(z_max, np.max(z_coords))

        # Return the minimum and maximum coordinates for each axis
        return x_min, x_max, y_min, y_max, z_min, z_max

    # PyTest exist
    @staticmethod
    def find_valid_space(df: pd.DataFrame) -> tuple[np.array, float, float, float]:
        ''' 
        This function takes in a Pandas DataFrame containing the transformed bounding box specifications of one or more 3D objects and calculates the expanded bounding box containing all of the 3D objects. 
        Args:
            df: A Pandas DataFrame containing the transformed bounding box specifications of one or more 3D objects. The DataFrame should have the columns "X-Min_transf", "X-Max_transf", "Y-Min_transf", "Y-Max_transf", "Z-Min_transf", and "Z-Max_transf" defining the bounding box specifications of each object. 
        Return: 
            A tuple containing the corner matrix, length, width, and height of the expanded bounding box. The corner matrix is a numpy array of eight corner points, each represented as a tuple of (x, y, z) coordinates. The length, width, and height are represented as float values. 
        '''

        # Initialize the minimum and maximum coordinates with the values from the first row of the DataFrame
        x_min = df.loc[0, 'X-Min_transf']
        x_max = df.loc[0, 'X-Max_transf']
        y_min = df.loc[0, 'Y-Min_transf']
        y_max = df.loc[0, 'Y-Max_transf']
        z_min = df.loc[0, 'Z-Min_transf']
        z_max = df.loc[0, 'Z-Max_transf']

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            # Update the minimum and maximum coordinates for each axis if necessary
            if row['X-Min_transf'] < x_min:
                x_min = row['X-Min_transf']
            if row['X-Max_transf'] > x_max:
                x_max = row['X-Max_transf']
            if row['Y-Min_transf'] < y_min:
                y_min = row['Y-Min_transf']
            if row['Y-Max_transf'] > y_max:
                y_max = row['Y-Max_transf']
            if row['Z-Min_transf'] < z_min:
                z_min = row['Z-Min_transf']
            if row['Z-Max_transf'] > z_max:
                z_max = row['Z-Max_transf']

        # Add 10% to each side to expand the bounding box
        expand_box_percent = 0.10
        add_length = (x_max - x_min) * expand_box_percent
        add_width = (y_max - y_min) * expand_box_percent
        add_height = (z_max - z_min) * expand_box_percent
        x_min = x_min - add_length
        x_max = x_max + add_length
        y_min = y_min - add_width
        y_max = y_max + add_width
        z_min = z_min - add_height
        z_max = z_max + add_height

        # Calculate the length, width, and height of the expanded bounding box
        length = x_max - x_min 
        width = y_max - y_min
        height = z_max - z_min

        # Define the corner matrix using the expanded bounding box coordinates
        corners = np.array([[x_min, y_min, z_min],
                    [x_min, y_min, z_max],
                    [x_min, y_max, z_min],
                    [x_min, y_max, z_max],
                    [x_max, y_min, z_min],
                    [x_max, y_min, z_max],
                    [x_max, y_max, z_min],
                    [x_max, y_max, z_max]])

        # Return the corner matrix, length, width, and height of the expanded bounding box
        return corners, length, width, height

    # PyTest exist
    @staticmethod
    def random_centerpoint_in_valid_space(corners: list, length: float, width: float, height: float) -> np.array:
        ''' 
        This function generates a random center point within the valid space defined by a 3D object's bounding box. 
        Args:
            corners: A list of eight corner points of the transformed bounding box in 3D space. Each point is a tuple of (x, y, z) coordinates.
            length: The length of the 3D object along the x-axis.
            width: The width of the 3D object along the y-axis.
            height: The height of the 3D object along the z-axis. 
        Return: 
            A numpy array containing three float values representing the random center point within the valid space of the 3D object, along the x, y, and z axes, respectively. 
        '''
        # Extract the minimum and maximum coordinates from the corners
        x_min, y_min, z_min = corners[0]
        x_max, y_max, z_max = corners[7]

        # Adjust the minimum and maximum coordinates by half of the dimensions
        x_min = x_min + (length / 2)
        x_max = x_max - (length / 2)
        y_min = y_min + (width / 2)
        y_max = y_max - (width / 2)
        z_min = z_min + (height / 2)
        z_max = z_max - (height / 2)

        # Generate random coordinates within the adjusted minimum and maximum ranges
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z = np.random.uniform(z_min, z_max)

        # Return the random center point as a numpy array
        return np.array([x, y, z])

    # PyTest exist
    @staticmethod
    def calculate_center_point(transformed_boundingbox: np.array) -> float:
        ''' 
        This function takes in the transformed bounding box as a list of 8 corner points in 3D space and computes the center point of the box. 
        Args: 
            transformed_boundingbox: A array of eight corner points of the transformed bounding box in 3D space. Each point is a tuple of (x, y, z) coordinates. 
        Return: center_x, center_y, center_z -> Three float values representing the center point of the bounding box along the x, y, and z axes, respectively. 
        '''

        # Initialize variables to store the sum of X, Y, and Z coordinates
        sum_X = 0
        sum_Y = 0
        sum_Z = 0

        # Get the number of corners in the transformed bounding box
        num_corners = len(transformed_boundingbox)

        # Iterate over each corner in the transformed bounding box
        for xyz in transformed_boundingbox:
            # Add the X, Y, and Z coordinates to the respective sums
            sum_X = sum_X + xyz[0]
            sum_Y = sum_Y + xyz[1]
            sum_Z = sum_Z + xyz[2]
        
        # Calculate the center coordinates by dividing the sums by the number of corners
        center_x = sum_X / num_corners
        center_y = sum_Y / num_corners
        center_z = sum_Z / num_corners

        # Return the center coordinates
        return center_x, center_y, center_z

    # PyTest exist
    @staticmethod
    def calculate_lwh(transformed_boundingbox: list) -> float:
        ''' 
        This function takes in the transformed bounding box as a list of 8 corner points in 3D space and computes the length, width, and height of the bounding box. 
        Args: 
            transformed_boundingbox: A list of eight corner points of the transformed bounding box in 3D space. 
            Each point is a tuple of (x, y, z) coordinates. 
        Return: Three float values representing the length, width, and height of the bounding box, respectively. 
        '''

        # Initialize empty lists to store the X, Y, and Z coordinates
        x = []
        y = []
        z = []

        # Iterate over each corner in the transformed bounding box
        for xyz in transformed_boundingbox:
            # Append the X, Y, and Z coordinates to their respective lists
            x.append(xyz[0])  
            y.append(xyz[1])  
            z.append(xyz[2])   

        # Calculate the length, width, and height by finding the difference between the maximum and minimum coordinates
        length = max(x) - min(x) 
        width = max(y) - min(y) 
        height = max(z) - min(z) 
        
        # Return the length, width, and height
        return length, width, height

    # PyTest exist
    @staticmethod
    def rotation_to_orientation(rot_mat: np.array) -> tuple[float, float, float]:
        ''' 
        Calculates the orientation angles (in radians) of a given rotation matrix in 3D space.
        Args:
            rotation matrix: an 3x3 matrix
        Return:
            theta_x: orientation angle of largest eigenvector with respect to x axis
            theta_y: orientation angle of largest eigenvector with respect to y axis
            theta_z: orientation angle of largest eigenvector with respect to z axis 
        '''
        if round(rot_mat[2,0],1) != -1.0 and round(rot_mat[2,0],1) != 1.0:
            theta_y = -np.arcsin(rot_mat[2,0])
            theta_x = np.arctan2(rot_mat[2,1]/np.cos(theta_y), rot_mat[2,2]/np.cos(theta_y))
            theta_z = np.arctan2(rot_mat[1,0]/np.cos(theta_y), rot_mat[0,0]/np.cos(theta_y))
        else:
            theta_z = 0
            if round(rot_mat[2,0],1) == -1.0:
                theta_y = math.pi/2
                theta_x = np.arctan2(rot_mat[0,1], rot_mat[0,2])
            else:
                theta_y = -math.pi/2
                theta_x = np.arctan2(-rot_mat[0,1], -rot_mat[0,2])

        return theta_x, theta_y, theta_z

    # PyTest exist
    @staticmethod
    def calculate_transformed_corners(center_point: np.array, length: float, width: float, height: float, theta_x: float, theta_y: float, theta_z: float) -> float: 
        ''' 
        This function takes in the location, dimensions, and orientation of a 3D object and calculates the coordinates of its eight corners. 
        It takes in seven arguments and returns six outputs: the minimum and maximum x, y, and z coordinates of the bounding box of the 3D object. 
        Args:
            center_point: A tuple containing the (x, y, z) coordinates of the center of the 3D object.
            length: The length of the 3D object along the x-axis.
            width: The width of the 3D object along the y-axis.
            height: The height of the 3D object along the z-axis.
            theta_x: The rotation of the 3D object around the x-axis, measured in radians.
            theta_y: The rotation of the 3D object around the y-axis, measured in radians.
            theta_z: The rotation of the 3D object around the z-axis, measured in radians. 
        Return: Six float values representing the minimum and maximum x, y, and z coordinates of the bounding box of the 3D object. 
        '''
        # Calculate the rotation matrix using the Euler angles
        
        rotation_matrix = np.array([
            [math.cos(float(theta_y)) * math.cos(float(theta_z)), -math.cos(float(theta_y)) * math.sin(float(theta_z)), math.sin(float(theta_y))],
            [math.cos(float(theta_x)) * math.sin(float(theta_z)) + math.sin(float(theta_x)) * math.sin(float(theta_y)) * math.cos(float(theta_z)), math.cos(float(theta_x)) * math.cos(float(theta_z)) - math.sin(float(theta_x)) * math.sin(float(theta_y)) * math.sin(float(theta_z)), -math.sin(float(theta_x)) * math.cos(float(theta_y))],
            [math.sin(float(theta_x)) * math.sin(float(theta_z)) - math.cos(float(theta_x)) * math.sin(float(theta_y)) * math.cos(float(theta_z)), math.sin(float(theta_x)) * math.cos(float(theta_z)) + math.cos(float(theta_x)) * math.sin(float(theta_y)) * math.sin(float(theta_z)), math.cos(float(theta_x)) * math.cos(float(theta_y))]
        ])
        # Calculate the half-lengths of the box along each axis
        half_length = length / 2
        half_width = width / 2
        half_height = height / 2
        # Calculate the coordinates of the eight corners of the box
        corners = np.array([
            [half_length, -half_width, -half_height],
            [half_length, -half_width, half_height],
            [half_length, half_width, -half_height],
            [half_length, half_width, half_height],
            [-half_length, -half_width, -half_height],
            [-half_length, -half_width, half_height],
            [-half_length, half_width, -half_height],
            [-half_length, half_width, half_height]
        ])
        # Rotate the corners using the rotation matrix
        rotated_corners = np.dot(corners, rotation_matrix)
        # Translate the corners to the center point
        translated_corners = rotated_corners + np.array([center_point[0], center_point[1], center_point[2]])
        # Calculate the minimum and maximum x, y, and z coordinates of the new bounding box
        x_min = np.min(translated_corners[:, 0])
        x_max = np.max(translated_corners[:, 0])
        y_min = np.min(translated_corners[:, 1])
        y_max = np.max(translated_corners[:, 1])
        z_min = np.min(translated_corners[:, 2])
        z_max = np.max(translated_corners[:, 2])

        return x_min, x_max, y_min, y_max, z_min, z_max

    @staticmethod
    def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
        '''
        The function takes a pandas DataFrame as input and adds new features/variables by calculating the bounding box coordinates, orientation, center point, length, width, height, volume, and density for each car part in the DataFrame. 
        It returns the updated pandas DataFrame with the new features/variables added. 
        Args: 
            dataframe: A pandas DataFrame object. 
        Return: 
            df: A pandas DataFrame object with the new features/variables added.
        '''
        # If X-Max eqauls 10000 it means that there are no bounding box information available. Therefore these samples are initialized to 0.
        df.loc[df['X-Max'] == 10000, ['X-Min', 'X-Max', 'Y-Min', 'Y-Max', 'Z-Min', 'Z-Max', 'ox', 'oy', 'oz', 'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz', 'Wert']] = 0

        for index, row in df.iterrows():  
            # Calculate and add new features to represent the bounding boxes
            transformed_boundingbox, rotation_matrix = Feature_Engineering.transform_boundingbox(row['X-Min'], row['X-Max'], row['Y-Min'], row['Y-Max'], row['Z-Min'], row['Z-Max'],row['ox'],row['oy'],row['oz'],row['xx'],row['xy'],row['xz'],row['yx'],row['yy'],row['yz'],row['zx'],row['zy'],row['zz'])
            center_x, center_y, center_z = Feature_Engineering.calculate_center_point(transformed_boundingbox)
            length, width, height = Feature_Engineering.calculate_lwh(transformed_boundingbox=transformed_boundingbox)
            theta_x, theta_y, theta_z = Feature_Engineering.rotation_to_orientation(rotation_matrix)
            x_coords = transformed_boundingbox[:, 0]
            y_coords = transformed_boundingbox[:, 1]
            z_coords = transformed_boundingbox[:, 2]

            df.at[index, 'X-Min_transf'] = min(x_coords)
            df.at[index, 'X-Max_transf'] = max(x_coords)
            df.at[index, 'Y-Min_transf'] = min(y_coords)
            df.at[index, 'Y-Max_transf'] = max(y_coords)
            df.at[index, 'Z-Min_transf'] = min(z_coords)
            df.at[index, 'Z-Max_transf'] = max(z_coords)   
            df.at[index, 'center_x'] = center_x
            df.at[index, 'center_y'] = center_y
            df.at[index, 'center_z'] = center_z
            df.at[index, 'length'] = length
            df.at[index, 'width'] = width
            df.at[index, 'height'] = height
            df.at[index, 'theta_x'] = theta_x
            df.at[index, 'theta_y'] = theta_y
            df.at[index, 'theta_z'] = theta_z

            # Calculate and add the volume as new feature 
            volume = length * width * height
            df.at[index, 'volume'] = volume

            # If weight is availabe, calculate and add the density as new feature 
            if pd.notnull(row['Wert']) and volume != 0:
                density = row['Wert'] / volume
                df.at[index, 'density'] = density
            
        df.loc[df['Wert'].isnull(), ['Wert']] = 0
        df.loc[df['density'].isnull(), ['density']] = 0
            
        return df