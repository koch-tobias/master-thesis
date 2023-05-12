lgbm_params = {
    "seed": 42,
    "cut_percent_of_front": 0.25,
    "test_size": 0.4,
    "metrics": ['auc', 'binary_logloss'],
     "n_estimators": 500,
    "early_stopping": 30
}

lgbm_hyperparameter = {
    "lr": [0.05],    
    "num_leaves": [31],
    "feature_fraction": [0.9],
}