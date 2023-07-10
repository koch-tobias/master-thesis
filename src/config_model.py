general_params = {
    "seed": 42,
    "cut_percent_of_front": 0.20,
    "relevant_features": ['Sachnummer','Benennung (dt)', 'X-Min','X-Max','Y-Min','Y-Max','Z-Min','Z-Max', 'Wert','Einheit','Gewichtsart','Kurzname','L-Kz.', 'L/R-Kz.', 'Modul (Nr)', 'ox','oy', 'oz', 'xx','xy','xz', 'yx','yy','yz','zx','zy','zz'],
    "features_for_model": ['volume', 'Wert', 'center_x', 'center_y', 'center_z','length','width','height','theta_x','theta_y','theta_z'],
    "keep_modules": ['CE05', 'CD07'], # Keep this modules out of module group EP (CE05=interior mirror, CD07=roof antenna)
    "car_part_designation": "Benennung (dt)"
}

paths = {
    "labeled_data": "data/labeled",
    "new_data": "data/raw_for_labeling",
    "model_folder": "models/HyperparameterTuning_07072023_0936"
}

gpt_settings = {
    "temperature": 0.6,
    "max_tokens": 200,
    "top_p": 1,
    "n": 1
}

train_settings = {
    "k-folds": 3,
    "augmentation": True,
    "use_only_text": False,
    "early_stopping": 100,
    "test_size": 0.3
    }

lgbm_params_binary = {
    "boosting_type": 'gbdt',
    "metrics": ['auc', 'binary_logloss'],
     "n_estimators": 10000,
    "prediction_threshold": 0.75
}

lgbm_params_multiclass = {
    "boosting_type": 'gbdt',
    "metrics": ['auc_mu', 'multi_logloss'],
     "n_estimators": 10000
    }

lgbm_hyperparameter = {
    "lr": [0.05, 0.07, 0.1],    
    "max_depth": [3, 5, 7],
    "colsample_bytree": [0.5, 0.7, 0.9],
    "min_child_samples": [20, 30, 40]
}

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