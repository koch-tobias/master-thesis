import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

import math
import re
import pickle

def transform_boundingbox(x_min, x_max, y_min, y_max, z_min, z_max, ox, oy, oz, xx, xy, xz, yx, yy, yz, zx, zy, zz):
    '''
    This function transforms a bounding box by applying rotation and translation.
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

    # Return the transformed corner points
    return transformed_corners


# %%
def get_minmax(list_transformed_bboxes):
    '''
    This function calculates the minimum and maximum coordinates of a list of transformed bounding boxes.
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


# %%
def find_valid_space(df):
    '''
    This function finds the valid space by calculating the minimum and maximum coordinates of a DataFrame and expanding it by 10%.
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


# %%
def random_centerpoint_in_valid_space(corners, length, width, height):
    '''
    This function generates a random center point within the valid space defined by the corners and dimensions.
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


# %%
def calculate_center_point(transformed_boundingbox):
    '''
    This function calculates the center point of a transformed bounding box.
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


# %%
def calculate_lwh(transformed_boundingbox):
    '''
    This function calculates the length, width, and height of a transformed bounding box.
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

def calculate_orientation(transformed_boundingbox):
    # Center the corners around the origin
    centered_corners = transformed_boundingbox - np.mean(transformed_boundingbox, axis=0) 

    # The function uses Singular Value Decomposition (SVD) to find the principal axes of the bounding box. SVD is a procedure that decomposes a matrix into three matrices: an orthogonal matrix U, a diagonal matrix S, and another orthogonal matrix V. The diagonal entries of S are the singular values of the matrix, which represent the size of the principal axes.
    u, s, principal_axes = np.linalg.svd(centered_corners)

    # Convert the principal axes to Euler angles
    theta_x = np.arctan2(principal_axes[2, 1], principal_axes[2, 2])
    theta_y = np.arctan2(-principal_axes[2, 0], np.sqrt(principal_axes[2, 1]**2 + principal_axes[2, 2]**2))
    theta_z = np.arctan2(principal_axes[1, 0], principal_axes[0, 0])

    return theta_x, theta_y, theta_z

# %%
def calculate_corners(center_point, length, width, height, theta_x, theta_y, theta_z): 
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

# %%
def prepare_text(designation: str) -> str:
    # transform to lower case
    text = str(designation).upper()

    # Removing punctations
    text = re.sub(r"[^\w\s]", " ", text)

    # Removing numbers
    text = ''.join([i for i in text if not i.isdigit()])

    # tokenize text
    text = text.split(" ")

    # Remove predefined words
    predefined_words = ["ZB", "AF", "LI", "RE", "MD", "LL", "TAB", "TB"]
    text = [word for word in text if word not in predefined_words]

    # Remove words with only one letter
    text = [word for word in text if len(word) > 1]

    # remove empty tokens
    text = [word for word in text if len(word) > 0]

    # join all
    prepared_designation = " ".join(text)

    return prepared_designation

# %%
def clean_text(df):
    df["Benennung (bereinigt)"] = df.apply(lambda x: prepare_text(x["Benennung (dt)"]), axis=1)

    return df
'''
def tokenize(text):
    return text.split()

def doc2vec_text_to_vec(data: pd.DataFrame, model_folder_path):

    # Tokenisierung der Bezeichnungen
    tokenized_texts = [tokenize(text) for text in data['Benennung (bereinigt)']]

    # Erstellen von TaggedDocuments
    tagged_data = [TaggedDocument(words=words, tags=[idx]) for idx, words in enumerate(tokenized_texts)]

    # Training des Doc2Vec-Modells
    vectorizer = Doc2Vec(tagged_data, vector_size=2000, min_count=1, epochs=100)

    # Vokabular extrahieren
    #vocabulary = vectorizer.wv.index_to_key
    vocabulary = vectorizer.build_vocab(tagged_data)

    # Vektorisierung der Sätze
    vectors = []
    for sentence in data['Benennung (bereinigt)']:
        sentence_vector = vectorizer.infer_vector(sentence.split())
        vectors.append(sentence_vector)

    X_text = np.array(vectors)

    with open(model_folder_path + 'vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(model_folder_path + 'vocabulary.pkl', 'wb') as f:
        pickle.dump(vocabulary, f)

    return X_text

def bert_text_to_vec(data: pd.DataFrame, model_folder_path):
    # Laden des vortrainierten deutschen BERT-Modells
    model_name = 'bert-base-german-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Tokenisierung der Bezeichnungen
    tokenized_texts = [tokenizer.tokenize(text) for text in data['Benennung (bereinigt)']]

    # Vektorisierung der Sätze
    vectors = []
    for sentence in tokenized_texts:
        # Konvertierung der Tokens in IDs
        input_ids = tokenizer.convert_tokens_to_ids(sentence)
        # Hinzufügen der Spezialtokens
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
        with model.as_default():
            outputs = model(input_ids)
            embeddings = outputs[0][:, 0, :].numpy()
            vectors.append(embeddings)

    X_text = np.array(vectors)

    with open(model_folder_path + 'tokenizer.pkl', 'wb') as f:
        pickle.dump(model, f)

    return X_text
'''
# %%
def nchar_text_to_vec(data: pd.DataFrame, model_folder_path: str) -> tuple:
    '''
    This function converts text data into vector representation using the n-gram approach.

    Args:
        data (pd.DataFrame): The input DataFrame containing the text data.
        model_folder_path (str): The path to the folder where the model files will be saved.

    Returns:
        tuple: A tuple containing the vectorized text data.
    '''

    # Initialize the CountVectorizer with the desired settings
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 8), max_features=8000)

    # Convert the text data into a vector representation
    X_text = vectorizer.fit_transform(data['Benennung (bereinigt)']).toarray()

    # Store the vocabulary
    vocabulary = vectorizer.get_feature_names_out()

    # Save the vectorizer and vocabulary if a model folder path is provided
    if model_folder_path != "":
        with open(model_folder_path + 'vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(model_folder_path + 'vocabulary.pkl', 'wb') as f:
            pickle.dump(vocabulary, f)

    # Return the vectorized text data
    return X_text


# %%
def get_vocabulary(column):
    '''
    This function extracts the vocabulary from a given column of text data.

    Args:
        column: The input column containing the text data.

    Returns:
        list: A list of unique words in the text data.
    '''

    # Concatenate all the text data into a single string
    text = ' '.join(column.astype(str))

    # Split the text into individual words and convert them to uppercase
    words = text.upper().split()

    # Count the occurrences of each word and sort them in descending order
    word_counts = pd.Series(words).value_counts()

    # Extract the unique words as the vocabulary
    vocabulary = word_counts.index.tolist()

    # Return the vocabulary
    return vocabulary

# %%
def calculate_orientation2(transformed_boundingbox):
    # Calculate the centroid of the bounding box
    centroid = np.mean(transformed_boundingbox, axis=0)

    # Calculate the differences between each corner and the centroid
    differences = transformed_boundingbox - centroid

    # Calculate the covariance matrix of the differences
    covariance_matrix = np.cov(differences, rowvar=False)

    # Perform eigenvalue decomposition on the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Extract the eigenvector corresponding to the largest eigenvalue
    largest_eigenvector = eigenvectors[:, 0]

    # Calculate the orientation angles
    theta_x = np.arctan2(largest_eigenvector[2], largest_eigenvector[1])
    theta_y = np.arctan2(largest_eigenvector[0], largest_eigenvector[2])
    theta_z = np.arctan2(largest_eigenvector[1], largest_eigenvector[0])

    return theta_x, theta_y, theta_z

import matplotlib.pyplot as plt
def plot_box(transformed_box):
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')

   # Unpack the coordinates from the transformed box
   x = [point[0] for point in transformed_box]
   y = [point[1] for point in transformed_box]
   z = [point[2] for point in transformed_box]

   # Plot the box
   ax.scatter(x, y, z)
   ax.set_xlabel('X Label')
   ax.set_ylabel('Y Label')
   ax.set_zlabel('Z Label')
   # Set axis limits
   ax.set_xlim(-2000, 5000)
   ax.set_ylim(-1500, 1500)
   ax.set_zlim(-100, 1500)
   
   plt.show()

# %%
def main():
    # Define input DataFrame    
    transformed_boundingbox = np.array([[3187.35199114, -102.46603984,  333.48725461],
                                        [3169.54398131, -102.46604762,  363.38566237],
                                        [3187.35199114,  102.53396016,  333.48730043],
                                        [3169.54398131,  102.53395238,  363.38570818],
                                        [3267.95969216, -102.46603984,  381.49859835],
                                        [3250.15168233, -102.46604762,  411.39700611],
                                        [3267.95969216,  102.53396016,  381.49864417],
                                        [3250.15168233,  102.53395238,  411.39705192]])
    
    #plot_box(transformed_boundingbox)
    # Call the function
    #result = calculate_orientation2(transformed_boundingbox)
    #result2 = calculate_orientation(transformed_boundingbox)

    #print(result)
    #print(result2)

    input_text = "This is a test A.F.-51232/AF designation." 
    print(input_text)
    result = prepare_text(input_text)
    print(result)
    
# %%
if __name__ == "__main__":
    
    main()