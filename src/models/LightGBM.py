# Used to define the model architectures

# %%
from lightgbm import LGBMClassifier
from configs.config_model import lgbm_params_multiclass, lgbm_params_binary
from configs.config_model import lgbm_hyperparameter as lgbm_hp

# %%
def binary_classifier(weight_factor, lr_index, max_depth_index, feature_frac_index, child_index):
    
    class_weight = {0: 1, 1: weight_factor}
    gbm = LGBMClassifier(boosting_type=lgbm_params_binary["boosting_type"],
                        objective='binary',
                        metric=lgbm_params_binary["metrics"],
                        num_leaves= pow(2, lgbm_hp["max_depth"][max_depth_index]),
                        max_depth=lgbm_hp["max_depth"][max_depth_index],
                        learning_rate=lgbm_hp['lr'][lr_index],
                        feature_fraction=lgbm_hp["feature_fraction"][feature_frac_index],
                        min_child_samples=lgbm_hp["min_child_samples"][child_index],
                        n_estimators=lgbm_params_binary["n_estimators"],
                        class_weight=class_weight)
    return gbm, lgbm_params_binary["metrics"]
    
def multiclass_classifier(weight_factor, lr_index, max_depth_index, feature_frac_index, child_index):
    gbm = LGBMClassifier(boosting_type=lgbm_params_multiclass["boosting_type"],
                        objective='multiclass',
                        metric=lgbm_params_multiclass["metrics"],
                        num_leaves= pow(2, lgbm_hp["max_depth"][max_depth_index]),
                        max_depth=lgbm_hp["max_depth"][max_depth_index],
                        learning_rate=lgbm_hp['lr'][lr_index],
                        feature_fraction=lgbm_hp["feature_fraction"][feature_frac_index],
                        min_child_samples=lgbm_hp["min_child_samples"][child_index],
                        n_estimators=lgbm_params_multiclass["n_estimators"],
                        num_class=len(weight_factor),
                        class_weight=weight_factor
                        )

    return gbm, lgbm_params_multiclass["metrics"]    
