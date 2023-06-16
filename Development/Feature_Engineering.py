# %%
import pandas as pd
from loguru import logger
from boundingbox_calculations import transform_boundingbox, calculate_center_point, calculate_lwh, calculate_orientation
from plot_functions import plot_vehicle
from text_preprocessing import clean_text

# %%
def add_new_features(df):
    for index, row in df.iterrows():  
        # Calculate and add new features to represent the bounding boxes
        transformed_boundingbox = transform_boundingbox(row['X-Min'], row['X-Max'], row['Y-Min'], row['Y-Max'], row['Z-Min'], row['Z-Max'],row['ox'],row['oy'],row['oz'],row['xx'],row['xy'],row['xz'],row['yx'],row['yy'],row['yz'],row['zx'],row['zy'],row['zz'])
        center_x, center_y, center_z = calculate_center_point(transformed_boundingbox)
        length, width, height = calculate_lwh(transformed_boundingbox)
        theta_x, theta_y, theta_z = calculate_orientation(transformed_boundingbox)

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

# %%
def preprocess_dataset(df, cut_percent_of_front: float):
    logger.info(f"Start preprocessing the dataframe with {df.shape[0]} samples...")

    df.loc[df['X-Max'] == 10000, ['X-Min', 'X-Max', 'Y-Min', 'Y-Max', 'Z-Min', 'Z-Max', 'ox', 'oy', 'oz', 'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz', 'Wert']] = 0

    df_new_features = add_new_features(df)

    # Using dictionary to convert specific columns
    convert_dict = {'X-Min': float,
                    'X-Max': float,
                    'Y-Min': float,
                    'Y-Max': float,
                    'Z-Min': float,
                    'Z-Max': float,
                    'Wert': float,
                    'ox': float,
                    'oy': float,
                    'oz': float,
                    'xx': float,
                    'xy': float,
                    'xz': float,
                    'yx': float,
                    'yy': float,
                    'yz': float,
                    'zx': float,
                    'zy': float,
                    'zz': float                     
                    }
    
    df_new_features = df_new_features.astype(convert_dict)

    df_new_features = df_new_features[(df_new_features['X-Min'] != 0) & (df_new_features['X-Max'] != 0)]

    # Save the samples without/wrong bounding box information in a new df, as they will need to be added back later
    df_temp = df[(df["X-Min"] == 0.0) & (df["X-Max"] == 0.0)]

    # Delete all samples which have less volume than 500,000 mm^3
    df_relevants = df_new_features[(df_new_features['volume'] > 500000)].reset_index(drop=True)

    # Delete all samples where the parts are in the front area of the car
    x_min_transf, x_max_transf = df_relevants["X-Min_transf"].min(), df_relevants["X-Max_transf"].max()
    car_length = x_max_transf - x_min_transf
    cut_point_x = x_min_transf + car_length*cut_percent_of_front
    df_relevants = df_relevants[df_relevants["X-Min_transf"] > cut_point_x]

    # Concatenate the two data frames vertically
    df_relevants = pd.concat([df_relevants, df_temp]).reset_index(drop=True)

    df_relevants = clean_text(df_relevants)

    # Drop the mirrored car parts (on the right sight) which have the same Sachnummer
    df_relevants = df_relevants[~((df_relevants['Sachnummer'].duplicated(keep='last')) & (df_relevants['yy'] == -1))]

    # Drop the mirrored car parts (on the right sight) which have not the same Sachnummer 
    df_relevants = df_relevants.loc[~(df_relevants.duplicated(subset='Kurzname', keep=False) & (df_relevants['L/R-Kz.'] == 'R'))]

    df_for_plot = df_relevants[(df_relevants['X-Min'] != 0) & (df_relevants['X-Max'] != 0)]

    #df_relevants = df_relevants.loc[:,["Sachnummer", "Benennung (bereinigt)", "X-Min_transf", "X-Max_transf", "Y-Min_transf", "Y-Max_transf", "Z-Min_transf", "Z-Max_transf", "center_x", "center_y", "center_z", "length", "width", "height", "theta_x", "theta_y", "theta_z", "volume", "density", "Benennung (dt)", "Wert", "Relevant fuer Messung", "Einheitsname"]]

    # Reset the index of the merged data frame
    df_relevants = df_relevants.reset_index(drop=True)

    logger.success(f"The dataset is successfully preprocessed. The new dataset contains {df_relevants.shape[0]} samples")

    return df_relevants, df_for_plot

# %% [markdown]
# # Main

# %%
def main(path, plot_preprocessed_data: bool):
    df = pd.read_excel(path,index_col=0) 

    if plot_preprocessed_data:
        df = df[(df['X-Max'] != 0) & (df['X-Min'] != 0)]
        df = df[df["Relevant fuer Messung"] == "Ja"]
        unique_names = df["Einheitsname"].unique().tolist()
        unique_names.sort()
        for name in unique_names:
            print(name)
            df_new = df[(df["Einheitsname"] == name)]
            plot_vehicle(df_new, add_valid_space=True, preprocessed_data=False, mirrored=False)
    else:
        df_preprocessed, df_for_plot = preprocess_dataset(df, cut_percent_of_front=0.20)
        plot_vehicle(df_for_plot, add_valid_space=True, preprocessed_data=False, mirrored=False)

# %%
if __name__ == "__main__":
    
    main(path="../data/artificial_dataset_05062023_1656.xlsx",plot_preprocessed_data=True)