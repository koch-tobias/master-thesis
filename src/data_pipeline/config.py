general_params = {
    "seed": 42,
    "cut_percent_of_front": 0.18,       # Choose number between 0 and 1. How much percent of the front of the car should be deleted - No relevant car parts in the front of a car
    "relevant_features": ['Sachnummer','Benennung (dt)', 'X-Min','X-Max','Y-Min','Y-Max','Z-Min','Z-Max', 'Wert','Einheit','Gewichtsart','Kurzname','L-Kz.', 'L/R-Kz.', 'Modul (Nr)', 'ox','oy', 'oz', 'xx','xy','xz', 'yx','yy','yz','zx','zy','zz'], # List of features which are relevant for the models - all other features will be deleted
    "features_for_model": ['volume', 'Wert', 'center_x', 'center_y', 'center_z','length','width','height','theta_x','theta_y','theta_z'], # List of features which are used for training the models
    "bounding_box_features_original": ['X-Min', 'X-Max', 'Y-Min', 'Y-Max', 'Z-Min', 'Z-Max', 'ox', 'oy', 'oz', 'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz', 'Wert'], #List of all original boundingbox features which should be set to 0 if wrong or not given
    "keep_modules": ['CE05', 'CD07'], # Keep this modules out of module group EP (CE05=interior mirror, CD07=roof antenna)
    "car_part_designation": "Benennung (dt)"
}

paths = {
    "labeled_data": "data/labeled",                                 # Paths to the labeld datasets
    "new_data": "data/raw_for_labeling"                             # Paths to the original datasets before labeling
}

gpt_settings = {
    "temperature": 0.6,            # Choose between 0 and 1. 1: gpt answers strict after guidlines and examples - 0: Fancy answers
    "max_tokens": 200,             # Maximal number of tokens which are alloud for you request
    "top_p": 1,                    #
    "n": 1                         # Number of answers gpt should return
}

