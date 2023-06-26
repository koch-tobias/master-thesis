general_params = {
    "seed": 42,
    "cut_percent_of_front": 0.20,
    "ncars": ['G11', 'G14', 'G15','G20', 'G22', 'G23', 'G29', 'G30', 'G61', 'G65', 'G70','NA5', 'NA7'],
    "relevant_features": ['Sachnummer','Benennung (dt)', 'X-Min','X-Max','Y-Min','Y-Max','Z-Min','Z-Max', 'Wert','Einheit','Gewichtsart','Kurzname','L-Kz.', 'L/R-Kz.', 'Modul (Nr)', 'ox','oy', 'oz', 'xx','xy','xz', 'yx','yy','yz','zx','zy','zz'],
    "features_for_model": ['center_x', 'center_y', 'center_z','length','width','height','theta_x','theta_y','theta_z'],
    "keep_modules": ['CE05', 'CD07'], # Keep this modules out of module group EP (CE05=interior mirror, CD07=roof antenna)
    "car_part_designation": "Benennung (dt)"
}

paths = {
    "labeled_data": "data/labeled_data",
    "new_data": "data/original_data_new",
}

paths_api = {
    "labeled_data": "../data/labeled_data",
    "new_data": "../data/original_data_new",
}

gpt_settings = {
    "temperature": 0.6,
    "max_tokens": 200,
    "top_p": 1,
    "n": 1
}

model_paths = {
    "lgbm_binary": "models/lgbm_07062023_1559",
    "lgbm_multiclass": "models/Einheitsnamen/lgbm_07062023_2130"
}

model_paths_api = {
    "lgbm_binary": "../models/lgbm_07062023_1559",
    "lgbm_multiclass": "../models/Einheitsnamen/lgbm_07062023_2130"
}

train_settings = {
    "cross_validation": False,
    "k-folds": 7,
    "augmentation": True,
    "use_only_text": False,
    "early_stopping": 30
}

lgbm_params_binary = {
    "boosting_type": 'dart',
    "test_size": 0.2,
    "metrics": ['auc', 'binary_logloss'],
     "n_estimators": 300,
    "prediction_threshold": 0.75
}

lgbm_params_multiclass = {
    "boosting_type": 'dart',
    "test_size": 0.2,
    "metrics": ['multi_logloss', 'auc_mu'],
     "n_estimators": 400
    }

lgbm_hyperparameter = {
    "lr": [0.05, 0.07, 0.1],    
    "max_depth": [3, 5, 7],
    "feature_fraction": [0.5, 0.7, 0.9],
    "min_child_samples": [20, 30, 40]
}