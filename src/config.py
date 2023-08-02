paths = {
    "folder_processed_dataset": "data/processed/31072023_1319/",            # Paths to the preprocessed dataset
    "final_model": "final_models"                                           # Paths to the directory of the final trained models
}

general_params = {
    "seed": 42,
    "cut_percent_of_front": 0.18,       # Choose number between 0 and 1. How much percent of the front of the car should be deleted - No relevant car parts in the front of a car
    "relevant_features": ['Sachnummer','Benennung (dt)', 'X-Min','X-Max','Y-Min','Y-Max','Z-Min','Z-Max', 'Wert','Einheit','Gewichtsart','Kurzname','L-Kz.', 'L/R-Kz.', 'Modul (Nr)', 'ox','oy', 'oz', 'xx','xy','xz', 'yx','yy','yz','zx','zy','zz'], # List of features which are relevant for the models - all other features will be deleted
    "features_for_model": ['volume', 'Wert', 'center_x', 'center_y', 'center_z','length','width','height','theta_x','theta_y','theta_z'], # List of features which are used for training the models
    "bounding_box_features_original": ['X-Min', 'X-Max', 'Y-Min', 'Y-Max', 'Z-Min', 'Z-Max', 'ox', 'oy', 'oz', 'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz', 'Wert'], #List of all original boundingbox features which should be set to 0 if wrong or not given
    "keep_modules": ['FBEP', 'CE05', 'CD07'], # Keep this modules out of module group EP (CE05=interior mirror, CD07=roof antenna)
    "car_part_designation": "Benennung (dt)",
    "augmentation": True,
    "use_only_text": False,         # True for training on only Designations or False for training on designations and bounding box information
}

prediction_settings = {
    "prediction_threshold": 0.65     # Model has to be x % sure that the car part is relevant
}

gpt_settings = {
    "temperature": 0.6,            # Choose between 0 and 1. 1: gpt answers strict after guidlines and examples - 0: Fancy answers
    "max_tokens": 200,             # Maximal number of tokens which are alloud for you request
    "top_p": 1,                    #
    "n": 1                         # Number of answers gpt should return
}

train_settings = {
    "k-folds": 4,                   # Number of folds for k-fold crossvalidation
    "ml-method": "lgbm",        # Choose between 'lgbm', 'xgboost' and 'catboost'
    "early_stopping": 50,           # Number of rounds for early stopping
    "train_val_split": 0.3,         # Split dataset into 1-x % training and x % validation
    "val_test_split": 0.5,          # Split validation set into 1-x % validation and x % test
    "top_x_models_for_cv": 0.10,    # Use the top x % of the models after hyperparamter tuning for cross validation
    "n_estimators": 10000          # Number of iterations a model will be trained
    }

###############################
lgbm_params_binary = {
    "boosting_type": 'gbdt',                # boosting type for lightgbm binary model
    "metrics": ['auc', 'binary_logloss']    # Area-under-the-curcve and logloss are used for training and evaluation
}

lgbm_params_multiclass = {
    "boosting_type": 'gbdt',                # boosting type for lightgbm binary model
    "metrics": ['auc_mu', 'multi_logloss']  # Area-under-the-curcve mu and logloss are used for training and evaluation
    }

lgbm_hyperparameter = {
    "lr": [0.05, 0.07, 0.1],                # Learning rate
    "max_depth": [3, 5, 7],                 # How long can be the distance from root to leafe node (to reduce overfitting)
    "colsample_bytree": [0.5, 0.7, 0.9],    # How much percent of the features are used in an iteration (to reduce overfitting)
    "min_child_samples": [20, 30, 40]       # Minimal number of datapoints a node must contain (to reduce overfitting)
}

###############################
xgb_params_binary = {
    "boosting_type": 'gbtree',              # Boosting type for Gradient-boosting decision tree
    "metrics": ['auc', 'logloss']            # Area-under-the-curve and loss
}

xgb_params_multiclass = {
    "boosting_type": 'gbtree',              # Boosting type for Gradient-boosting decision tree
    "metrics": ['auc', 'mlogloss'],   # Area-under-the-curve and multilogloss
    }

xgb_hyperparameter = {
    "lr": [0.05, 0.1, 0.3],                 # Learning rate
    "max_depth": [4, 6, 9],                 # How long can be the distance from root to leafe node (to reduce overfitting)
    "colsample_bytree": [0.5, 0.7, 0.9],    # How much percent of the features are used in an iteration (to reduce overfitting)
    "gamma": [0, 0.2, 0.5],                 # 

}

###############################
cb_params_binary = {
    "boosting_type": 'Plain',                # boosting type for lightgbm binary model
    "metrics": ['AUC', 'Logloss']            # Area-under-the-curcve and crossentropy are used for training and evaluation
}

cb_params_multiclass = {
    "boosting_type": 'Plain',                # boosting type for lightgbm binary model
    "metrics": ['AUC', 'MultiClass']  # Area-under-the-curcve mu and logloss are used for training and evaluation
    }

cb_hyperparameter = {
    "lr": [0.05, 0.07, 0.1],                # Learning rate
    "depth": [4, 6, 9],                     # How long can be the distance from root to leafe node (to reduce overfitting)
    "colsample_bylevel": [0.5, 0.7, 0.9],   # How much percent of the features are used in an iteration (to reduce overfitting) -> Not supported using gpu
    #'l2_leaf_reg':[1,5,10,100],
    "min_data_in_leaf": [20, 30, 40]        # Minimal number of datapoints a node must contain (to reduce overfitting)
}
################################

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
