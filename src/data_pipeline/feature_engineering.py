import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

import math
import re
import pickle

# %%
def transform_boundingbox(x_min, x_max, y_min, y_max, z_min, z_max, ox, oy, oz, xx, xy, xz, yx, yy, yz, zx, zy, zz):
    corners = np.array([[x_min, y_min, z_min],
                        [x_min, y_min, z_max],
                        [x_min, y_max, z_min],
                        [x_min, y_max, z_max],
                        [x_max, y_min, z_min],
                        [x_max, y_min, z_max],
                        [x_max, y_max, z_min],
                        [x_max, y_max, z_max]])
    
    rotation_matrix = np.array([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]])
    shift_vec = np.array([ox, oy, oz])
    rotated_corners = np.dot(corners, rotation_matrix) 
    transformed_corners = rotated_corners + shift_vec

    return transformed_corners 

# %%
def get_minmax(list_transformed_bboxes):
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    z_min = np.inf
    z_max = -np.inf

    for arr in list_transformed_bboxes:
        x_coords = arr[:, 0]
        y_coords = arr[:, 1]
        z_coords = arr[:, 2]
        
        x_min = min(x_min, np.min(x_coords))
        x_max = max(x_max, np.max(x_coords))
        y_min = min(y_min, np.min(y_coords))
        y_max = max(y_max, np.max(y_coords))
        z_min = min(z_min, np.min(z_coords))
        z_max = max(z_max, np.max(z_coords))

    return x_min, x_max, y_min, y_max, z_min, z_max

# %%
def find_valid_space(df):
    x_min = df.loc[0, 'X-Min_transf']
    x_max = df.loc[0, 'X-Max_transf']
    y_min = df.loc[0, 'Y-Min_transf']
    y_max = df.loc[0, 'Y-Max_transf']
    z_min = df.loc[0, 'Z-Min_transf']
    z_max = df.loc[0, 'Z-Max_transf']

    for index, row in df.iterrows():
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

    # Add 10% to each side
    expand_box_percent = 0.10
    add_length = (x_max - x_min) * expand_box_percent
    add_width = (y_max - y_min) * expand_box_percent
    add_height = (z_max - z_min) * expand_box_percent
    x_min = x_min-add_length
    x_max = x_max+add_length
    y_min = y_min-add_width
    y_max = y_max+add_width
    z_min = z_min-add_height
    z_max = z_max+add_height

    length = x_max - x_min 
    width = y_max - y_min
    height = z_max - z_min

    # Define the corner matrix
    corners = np.array([[x_min, y_min, z_min],
                [x_min, y_min, z_max],
                [x_min, y_max, z_min],
                [x_min, y_max, z_max],
                [x_max, y_min, z_min],
                [x_max, y_min, z_max],
                [x_max, y_max, z_min],
                [x_max, y_max, z_max]])

    return corners, length, width, height

# %%
def random_centerpoint_in_valid_space(corners, length, width, height):
    x_min, y_min, z_min = corners[0]
    x_max, y_max, z_max = corners[7]
    x_min = x_min + (length/2)
    x_max = x_max - (length/2)
    y_min = y_min + (width/2)
    y_max = y_max - (width/2)
    z_min = z_min + (height/2)
    z_max = z_max - (height/2)    
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    z = np.random.uniform(z_min, z_max)
    return np.array([x, y, z])

# %%
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

# %%
def calculate_lwh(transformed_boundingbox):
    x = []
    y = []
    z = []

    for xyz in transformed_boundingbox:
        x.append(xyz[0])  
        y.append(xyz[1])  
        z.append(xyz[2])   

    length = max(x) - min(x) 
    width = max(y) - min(y) 
    height = max(z) - min(z) 
    
    return length, width, height

# %%
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
        [math.cos(float(theta_y.iloc[0])) * math.cos(float(theta_z.iloc[0])), -math.cos(float(theta_y.iloc[0])) * math.sin(float(theta_z.iloc[0])), math.sin(float(theta_y.iloc[0]))],
        [math.cos(float(theta_x.iloc[0])) * math.sin(float(theta_z.iloc[0])) + math.sin(float(theta_x.iloc[0])) * math.sin(float(theta_y.iloc[0])) * math.cos(float(theta_z.iloc[0])), math.cos(float(theta_x.iloc[0])) * math.cos(float(theta_z.iloc[0])) - math.sin(float(theta_x.iloc[0])) * math.sin(float(theta_y.iloc[0])) * math.sin(float(theta_z.iloc[0])), -math.sin(float(theta_x.iloc[0])) * math.cos(float(theta_y.iloc[0]))],
        [math.sin(float(theta_x.iloc[0])) * math.sin(float(theta_z.iloc[0])) - math.cos(float(theta_x.iloc[0])) * math.sin(float(theta_y.iloc[0])) * math.cos(float(theta_z.iloc[0])), math.sin(float(theta_x.iloc[0])) * math.cos(float(theta_z.iloc[0])) + math.cos(float(theta_x.iloc[0])) * math.sin(float(theta_y.iloc[0])) * math.sin(float(theta_z.iloc[0])), math.cos(float(theta_x.iloc[0])) * math.cos(float(theta_y.iloc[0]))]
    ])

    # Calculate the half-lengths of the box along each axis
    half_length = length / 2
    half_width = width / 2
    half_height = height / 2

    # Calculate the coordinates of the eight corners of the box
    corners = np.array([
        [-half_length, -half_width, -half_height],
        [half_length, -half_width, -half_height],
        [half_length, half_width, -half_height],
        [-half_length, half_width, -half_height],
        [-half_length, -half_width, half_height],
        [half_length, -half_width, half_height],
        [half_length, half_width, half_height],
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
    text = re.sub(r"[^\w\s]", "", text)

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

def tokenize(text):
    return text.split()

def text_to_vec(data: pd.DataFrame, model_folder_path):

    # Tokenisierung der Bezeichnungen
    tokenized_texts = [tokenize(text) for text in data['Benennung (bereinigt)']]

    # Erstellen von TaggedDocuments
    tagged_data = [TaggedDocument(words=words, tags=[idx]) for idx, words in enumerate(tokenized_texts)]

    # Training des Doc2Vec-Modells
    vectorizer = Doc2Vec(tagged_data, vector_size=500, min_count=1, epochs=30)

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


# %%
def vectorize_data(data: pd.DataFrame, model_folder_path) -> tuple:

    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 8))

    X_text = vectorizer.fit_transform(data['Benennung (bereinigt)']).toarray()

    # Store the vocabulary
    vocabulary = vectorizer.get_feature_names_out()

    with open(model_folder_path + 'vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(model_folder_path + 'vocabulary.pkl', 'wb') as f:
        pickle.dump(vocabulary, f)

    return X_text

# %%
def get_vocabulary(column):
    text = ' '.join(column.astype(str))
    words = text.upper().split()
    word_counts = pd.Series(words).value_counts()
    vocabulary = word_counts.index.tolist()

    return vocabulary
