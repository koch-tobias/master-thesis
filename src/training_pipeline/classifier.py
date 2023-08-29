# %%
from lightgbm import LGBMClassifier
import xgboost as xgb
from catboost import CatBoostClassifier

import yaml
from yaml.loader import SafeLoader
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

class Classifier:

    @staticmethod
    def binary_classifier(weight_factor: int, hp: dict, method: str):
        '''
        This function is used to create a binary classifier model using either LightGBM, XGBoost or CatBoost libraries.
        Args:
            weight_factor: An integer value greater than 1 used to give more weight to the minority class during training
            hp: A dictionary containing different hyperparameters for the given model
            method: A string indicating the machine learning library to be used
        Returns:
            model: a binary classifier model object
            metrics: a list of strings representing the performance metrics to be used for training the model.
        '''
        
        class_weight = {0: 1, 1: weight_factor}
        if method == "lgbm":
            model = LGBMClassifier(boosting_type=config["lgbm_params_binary"]["boosting_type"],
                                    objective='binary',
                                    metric=config["lgbm_params_binary"]["metrics"],
                                    num_leaves= pow(2, hp["max_depth"]),
                                    max_depth=hp["max_depth"],
                                    learning_rate=hp["lr"],
                                    colsample_bytree=hp["colsample_bytree"],
                                    min_child_samples=hp["min_child_samples"],
                                    n_estimators=config["train_settings"]["n_estimators"] ,
                                    class_weight=class_weight
                                )
            metrics = config["lgbm_params_binary"]["metrics"]

        elif method == "xgboost":
            model = xgb.XGBClassifier(booster=config["xgb_params_binary"]["boosting_type"], 
                                        objective="binary:logistic",
                                        eval_metric = config["xgb_params_binary"]["metrics"],
                                        max_depth=hp["max_depth"],   
                                        learning_rate=hp["lr"],
                                        colsample_bytree=hp["colsample_bytree"],
                                        scale_pos_weight=weight_factor,
                                        gamma=hp["gamma"],
                                        n_estimators= config["train_settings"]["n_estimators"],
                                        #tree_method= 'gpu_hist',
                                        #predictor='gpu_predictor',
                                        #gpu_id=5,
                                        verbosity=0
                                    )        
            metrics = config["xgb_params_binary"]["metrics"]

        elif method == "catboost":
            model = CatBoostClassifier(iterations=config["train_settings"]["n_estimators"], 
                                        learning_rate=hp["lr"], 
                                        depth=hp["depth"],
                                        colsample_bylevel=hp["colsample_bylevel"],                          # Not supported by GPU
                                        min_data_in_leaf=hp["min_data_in_leaf"],
                                        loss_function=config["cb_params_binary"]["metrics"][1],
                                        eval_metric=config["cb_params_binary"]["metrics"][0],
                                        early_stopping_rounds=config["train_settings"]["early_stopping"],
                                        use_best_model=True
                                    )
            metrics = config["cb_params_binary"]["metrics"]
            
        return model, metrics
        
    @staticmethod
    def multiclass_classifier(weight_factor: dict, hp: dict, method: str):
        '''
        This function is used to create a multiclass classification model using either LightGBM, XGBoost or CatBoost libraries.
        Args:
            weight_factor: A dictionary containing the class weights used for training. This value is greater than or equal to one. It specifies the ratio of samples from each class weight. The keys of the dictionary represent the classes and the values specify the weights of each class.
            hp: A dictionary containing different hyperparameters for the given model.
            method: A string indicating the machine learning library to be used.
        Returns:
            model: a multiclass classifier model object.
            metrics: a list of strings representing the performance metrics to be used for training the model.
        '''
        if method == 'lgbm':
            model = LGBMClassifier(boosting_type=config["lgbm_params_multiclass"]["boosting_type"],
                                    objective='multiclass',
                                    metric=config["lgbm_params_multiclass"]["metrics"],
                                    num_leaves= pow(2, hp["max_depth"]),
                                    max_depth=hp["max_depth"],
                                    learning_rate=hp["lr"],
                                    colsample_bytree=hp["colsample_bytree"],
                                    min_child_samples=hp["min_child_samples"],
                                    n_estimators=config["train_settings"]["n_estimators"] ,
                                    num_class=len(weight_factor),
                                    class_weight=weight_factor
                                )
            metrics = config["lgbm_params_multiclass"]["metrics"]
            
        elif method == "xgboost":
            model = xgb.XGBClassifier(booster=config["xgb_params_multiclass"]["boosting_type"], 
                                        objective="multi:softprob",
                                        eval_metric = config["xgb_params_multiclass"]["metrics"],
                                        max_depth=hp["max_depth"],   
                                        learning_rate=hp["lr"],
                                        colsample_bytree=hp["colsample_bytree"],
                                        scale_pos_weight=weight_factor,
                                        gamma=hp["gamma"],
                                        n_estimators= config["train_settings"]["n_estimators"],
                                        #tree_method= 'gpu_hist',
                                        #predictor='gpu_predictor',
                                        #gpu_id=5,
                                        verbosity=0
                                    )   
            metrics = config["xgb_params_multiclass"]["metrics"]
        
        elif method == "catboost":
            model = CatBoostClassifier(iterations=config["train_settings"]["n_estimators"], 
                                        learning_rate=hp["lr"], 
                                        depth=hp["depth"],
                                        colsample_bylevel=hp["colsample_bylevel"],                      # Not supported by GPU
                                        min_data_in_leaf=hp["min_data_in_leaf"],
                                        loss_function=config["cb_params_multiclass"]["metrics"][1],
                                        eval_metric=config["cb_params_multiclass"]["metrics"][0],
                                        early_stopping_rounds=config["train_settings"]["early_stopping"],
                                        use_best_model=True
                                    )
            metrics = config["cb_params_multiclass"]["metrics"]

        return model, metrics