general_params = {
    "seed": 42,
    "cut_percent_of_front": 0.20,
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