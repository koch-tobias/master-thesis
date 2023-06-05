general_params = {
    "seed": 42,
    "cut_percent_of_front": 0.20,
    "save_preprocessed_data": False,
    "save_artificial_dataset": False,
    "save_prepared_dataset_for_labeling": False,
    "use_only_text": False,
    "ncars": ['G14', 'G15', 'G22', 'G23', 'G61', 'G65', 'NA5', 'NA7'],
    "relevant_features": ['Sachnummer','Benennung (dt)', 'X-Min','X-Max','Y-Min','Y-Max','Z-Min','Z-Max', 'Wert','Einheit','Gewichtsart','Kurzname','L-Kz.', 'L/R-Kz.', 'Modul (Nr)', 'ox','oy', 'oz', 'xx','xy','xz', 'yx','yy','yz','zx','zy','zz'],
    "features_for_model": ['center_x', 'center_y', 'center_z','length','width','height','theta_x','theta_y','theta_z'],
    "keep_modules": ['CE05', 'CD07'], # Keep this modules out of module group EP (CE05=interior mirror, CD07=roof antenna)
    "car_part_designation": "Benennung (dt)"
}

paths = {
    "labeled_data": "../data/labeled_data"
}

# using dictionary to convert data types of specific columns
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

lgbm_params = {
    "test_size": 0.4,
    "metrics": ['auc', 'binary_logloss'],
     "n_estimators": 1000,
    "early_stopping": 30
}

lgbm_hyperparameter = {
    "lr": [0.1],    
    "max_depth": [7],
    "num_leaves": [128],
    "feature_fraction": [0.9],
}