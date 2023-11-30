# %%
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer

import numpy as np

import yaml
from yaml.loader import SafeLoader
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

class CustomFbetaMetric():
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):

        y_pred = np.array(approxes).argmax(0)
            
        #y_pred = [np.argmax(x) for x in approxes]
        y_true = np.array(target)

        # Calculate the F-beta score for multiclass classification
        fbeta = fbeta_score(y_true, y_pred, beta=2, average='weighted')

        # Return the fbeta score
        return fbeta, 0

class Classifier:

    @staticmethod
    def costum_fbeta_score(y_true, y_pred):
        
        # Define the beta value
        beta = 2

        y_pred = np.round(y_pred)
        # Calculate the F-beta score with the specified beta value
        fbeta = fbeta_score(y_true, y_pred, beta=beta)

        return 'fbeta', fbeta, True

    @staticmethod
    def costum_fbeta_score_multi(y_true, y_pred):
        
        # Define the beta value
        beta = 2
        # Round the predicted probabilities to get the predicted labels
        y_pred_labels = np.argmax(y_pred, axis=1)
        
        # Calculate the F-beta score with the specified beta value
        fbeta = fbeta_score(y_true, y_pred_labels, beta=beta, average='weighted')
        
        return 'fbeta', fbeta, True
    
    def xgb_custom_fbeta_score(preds, dtrain):
        beta = 2  # Define the desired value for beta
        #labels = dtrain.get_label()  # Get the true labels from dtrain
        classes = np.repeat(0, dtrain.shape[0])
        classes[dtrain > 0.5] = 1
        y_pred = classes

        # Convert probability outputs to binary predictions
        preds_binary = np.round(preds)

        # Compute the F-beta score using sklearn's fbeta_score function
        fbeta = fbeta_score(y_pred, preds_binary, beta=beta)
        
        return fbeta

        
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
                                    metric='None',
                                    max_depth=hp["max_depth"],
                                    learning_rate=hp["lr"],
                                    colsample_bytree=hp["colsample_bytree"],
                                    reg_lambda=hp["lambda_l2"],
                                    n_estimators=config["train_settings"]["n_estimators"] ,
                                    class_weight=class_weight,
                                    random_state=42
                                )
            metrics = [config["lgbm_params_binary"]["loss"], Classifier.costum_fbeta_score]

        elif method == "xgboost":
            metrics =  Classifier.xgb_custom_fbeta_score

            model = XGBClassifier(booster=config["xgb_params_binary"]["boosting_type"], 
                                        objective="binary:logistic",
                                        eval_metric = metrics,
                                        max_depth=hp["max_depth"],   
                                        learning_rate=hp["lr"],
                                        colsample_bytree=hp["colsample_bytree"],
                                        reg_lambda=hp["lambda"],
                                        scale_pos_weight=weight_factor,
                                        n_estimators= config["train_settings"]["n_estimators"],
                                        verbosity=0,
                                        random_state=42
                                    )        

        elif method == "catboost":
            model = CatBoostClassifier(iterations=config["train_settings"]["n_estimators"], 
                                        learning_rate=hp["lr"], 
                                        depth=hp["depth"],
                                        l2_leaf_reg=hp["l2_regularization"],
                                        bagging_temperature=hp["bagging_temperature"],        
                                        loss_function=config["cb_params_binary"]["loss"],
                                        eval_metric='F:beta=2;use_weights=false',
                                        early_stopping_rounds=config["train_settings"]["early_stopping"],
                                        use_best_model=True,
                                        class_weights=class_weight,
                                        random_seed=42
                                    )
            metrics = [config["cb_params_binary"]["loss"], 'F:beta=2']
            
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
            metrics = config["lgbm_params_multiclass"]["loss"]
            model = LGBMClassifier(boosting_type=config["lgbm_params_multiclass"]["boosting_type"],
                                    objective='multiclass',
                                    metric=metrics,
                                    max_depth=hp["max_depth"],
                                    learning_rate=hp["lr"],
                                    colsample_bytree=hp["colsample_bytree"],
                                    reg_lambda=hp["lambda_l2"],
                                    min_child_samples=20,
                                    n_estimators=config["train_settings"]["n_estimators"] ,
                                    num_class=len(weight_factor),
                                    class_weight=weight_factor,
                                    random_state=42
                                )
            
        elif method == "xgboost":
            metrics = config["xgb_params_multiclass"]["loss"]
            model = XGBClassifier(booster=config["xgb_params_multiclass"]["boosting_type"], 
                                        objective="multi:softprob",
                                        eval_metric = metrics,
                                        max_depth=hp["max_depth"],   
                                        learning_rate=hp["lr"],
                                        colsample_bytree=hp["colsample_bytree"],
                                        scale_pos_weight=weight_factor,
                                        reg_lambda=hp["lambda"],
                                        n_estimators= config["train_settings"]["n_estimators"],
                                        verbosity=0,
                                        random_state=42
                                    )   
        
        elif method == "catboost":
            metrics = config["cb_params_multiclass"]["loss"]
            model = CatBoostClassifier(iterations=config["train_settings"]["n_estimators"], 
                                        learning_rate=hp["lr"], 
                                        depth=hp["depth"],
                                        l2_leaf_reg=hp["l2_regularization"],
                                        bagging_temperature=hp["bagging_temperature"],
                                        loss_function=config["cb_params_multiclass"]["loss"],
                                        eval_metric=config["cb_params_multiclass"]["loss"],
                                        early_stopping_rounds=config["train_settings"]["early_stopping"],
                                        class_weights=weight_factor,
                                        use_best_model=True,
                                        random_seed=42
                                    )

        return model, metrics