# %%
from lightgbm import LGBMClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from src.config import lgbm_params_multiclass, lgbm_params_binary, xgb_params_binary, xgb_params_multiclass, cb_params_binary, cb_params_multiclass, train_settings

# %%
def binary_classifier(weight_factor, hp, method):
    
    class_weight = {0: 1, 1: weight_factor}
    if method == "lgbm":
        model = LGBMClassifier(boosting_type=lgbm_params_binary["boosting_type"],
                            objective='binary',
                            metric=lgbm_params_binary["metrics"],
                            num_leaves= pow(2, hp["max_depth"]),
                            max_depth=hp["max_depth"],
                            learning_rate=hp["lr"],
                            colsample_bytree=hp["colsample_bytree"],
                            min_child_samples=hp["min_child_samples"],
                            n_estimators=train_settings["n_estimators"] ,
                            class_weight=class_weight
                            )
        metrics = lgbm_params_binary["metrics"]

    elif method == "xgboost":
        model = xgb.XGBClassifier(booster=xgb_params_binary["boosting_type"], 
                            objective="binary:logistic",
                            eval_metric = xgb_params_binary["metrics"],
                            max_depth=hp["max_depth"],   
                            learning_rate=hp["lr"],
                            colsample_bytree=hp["colsample_bytree"],
                            scale_pos_weight=weight_factor,
                            gamma=hp["gamma"],
                            n_estimators= train_settings["n_estimators"],
                            verbosity=0
                        )        
        metrics = xgb_params_binary["metrics"]

    elif method == "catboost":
        model = CatBoostClassifier(iterations=train_settings["n_estimators"], 
                                    learning_rate=hp["lr"], 
                                    depth=hp["depth"],
                                    colsample_bylevel=hp["colsample_bylevel"],
                                    min_data_in_leaf=hp["min_data_in_leaf"],
                                    loss_function=cb_params_binary["metrics"][1],
                                    eval_metric=cb_params_binary["metrics"][0],
                                    early_stopping_rounds=train_settings["early_stopping"],
                                    use_best_model=True
                                )
        metrics = cb_params_binary["metrics"]
        
    return model, metrics
    
    
def multiclass_classifier(weight_factor, hp, method):
    if method == 'lgbm':
        model = LGBMClassifier(boosting_type=lgbm_params_multiclass["boosting_type"],
                            objective='multiclass',
                            metric=lgbm_params_multiclass["metrics"],
                            num_leaves= pow(2, hp["max_depth"]),
                            max_depth=hp["max_depth"],
                            learning_rate=hp["lr"],
                            colsample_bytree=hp["colsample_bytree"],
                            min_child_samples=hp["min_child_samples"],
                            n_estimators=train_settings["n_estimators"] ,
                            num_class=len(weight_factor),
                            class_weight=weight_factor
                            #device='gpu'
                            )
        metrics = lgbm_params_multiclass["metrics"]
        
    elif method == "xgboost":
        model = xgb.XGBClassifier(booster=xgb_params_multiclass["boosting_type"], 
                            objective="multi:logistic",
                            max_depth=hp["max_depth"],   
                            learning_rate=hp["lr"],
                            colsample_bytree=hp["colsample_bytree"],
                            scale_pos_weight=weight_factor,
                            gamma=hp["gamma"],
                            n_estimators= train_settings["n_estimators"],
                            verbosity=2
                        )   
        metrics = xgb_params_multiclass["metrics"]
        

    return model, metrics     