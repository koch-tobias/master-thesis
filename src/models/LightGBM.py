# Used to define the model architectures

# %%
from lightgbm import LGBMClassifier
from config_model import lgbm_params_multiclass, lgbm_params_binary

# %%
def binary_classifier(weight_factor, lr, max_depth, colsample, child):
    
    class_weight = {0: 1, 1: weight_factor}
    gbm = LGBMClassifier(boosting_type=lgbm_params_binary["boosting_type"],
                        objective='binary',
                        metric=lgbm_params_binary["metrics"],
                        num_leaves= pow(2, max_depth),
                        max_depth=max_depth,
                        learning_rate=lr,
                        colsample_bytree=colsample,
                        min_child_samples=child,
                        n_estimators=lgbm_params_binary["n_estimators"],
                        class_weight=class_weight)
    return gbm, lgbm_params_binary["metrics"]
    
def multiclass_classifier(weight_factor, lr, max_depth, colsample, child):
    gbm = LGBMClassifier(boosting_type=lgbm_params_multiclass["boosting_type"],
                        objective='multiclass',
                        metric=lgbm_params_multiclass["metrics"],
                        num_leaves= pow(2, max_depth),
                        max_depth=max_depth,
                        learning_rate=lr,
                        colsample_bytree=colsample,
                        min_child_samples=child,
                        n_estimators=lgbm_params_multiclass["n_estimators"],
                        num_class=len(weight_factor),
                        class_weight=weight_factor
                        )

    return gbm, lgbm_params_multiclass["metrics"]    
