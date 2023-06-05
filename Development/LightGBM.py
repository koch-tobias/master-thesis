# %%
import numpy as np
import matplotlib.pyplot as plt

import pickle
from loguru import logger

import lightgbm as lgb
from lightgbm import LGBMClassifier
from lightgbm import plot_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import ConfusionMatrixDisplay

from Prepare_data import load_prepare_dataset
from config import lgbm_params
from config import lgbm_hyperparameter as lgbm_hp

# %%
def store_predictions(model, X_test, y_test, y_pred, probs, features, timestamp):
    vectorizer_path = f"../models/lgbm_{timestamp}/vectorizer.pkl"
    # Load the vectorizer from the file
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    vocabulary_path = f"../models/lgbm_{timestamp}/vocabulary.pkl"
    # Get the vocabulary of the training data
    with open(vocabulary_path, 'rb') as f:
        vocabulary = pickle.load(f)
        
    # Extrahieren der wichtigsten Features
    boost = model.booster_
    importance = boost.feature_importance()
    feature_names = boost.feature_name()
    sorted_idx = np.argsort(importance)[::-1]

    feature_dict = {vocabulary.shape[0]+index: key for index, key in enumerate(features)}

    true_label = y_test.reset_index(drop=True)

    X_test_restored = vectorizer.inverse_transform(X_test[:,:vocabulary.shape[0]-len(features)])
    original_designation = [' '.join(words) for words in X_test_restored]

    print('Wichtigsten Features:')
    for j in sorted_idx:
        if importance[j] > 100:
            if j < vocabulary.shape[0]:
                print('{} ({}) Value: {}'.format(feature_names[j], importance[j], vocabulary[j]))
            else:
                print('{} ({}) Value: {}'.format(feature_names[j], importance[j], feature_dict[j]))
        else:
            continue

    # Ausgabe der Vorhersagen, der Wahrscheinlichkeiten und der wichtigsten Features
    for i in range(len(X_test)):
        if y_pred[i] != true_label[i]:
            if y_pred[i] == 1:
                print('Vorhersage für Sample {}: Ja ({})'.format(i+1, y_pred[i]), 'True: Nein ({})'.format(true_label[i]))
            else:
                print('Vorhersage für Sample {}: Nein ({})'.format(i+1, y_pred[i]), 'True: Ja ({})'.format(true_label[i]))
            print(original_designation[i])

            print('Wahrscheinlichkeit für Sample {}: {}'.format(i+1, probs[i][1]))

            print('------------------------')



# %%
def train_model(X_train, y_train, X_val, y_val, weight_factor):
    
    class_weight = {0: 1, 1: weight_factor}
    evals = {}
    callbacks = [lgb.early_stopping(lgbm_params["early_stopping"]), lgb.record_evaluation(evals)]

    gbm = LGBMClassifier(boosting_type='dart',
                        objective='binary',
                        metric=['auc', 'binary_logloss'],
                        num_leaves=lgbm_hp["num_leaves"],
                        max_depth=lgbm_hp["max_depth"],
                        learning_rate=lgbm_hp['lr'],
                        feature_fraction=lgbm_hp["feature_fraction"],
                        n_estimators=lgbm_params["n_estimators"],
                        class_weight=class_weight)

    gbm.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)], 
            eval_metric=lgbm_params["metrics"],
            early_stopping_rounds=lgbm_params["early_stopping"],
            callbacks=callbacks)

    
    return gbm, evals

# %%
def store_trained_model(model, test_acc, timestamp):
    # save model
    model_path = f"../models/lgbm_{timestamp}/model_{str(test_acc)[2:]}.pkl"
    with open(model_path, "wb") as filestore:
        pickle.dump(model, filestore)

# %%
def evaluate_model(model, X_test, y_test, evals, timestamp):
    threshold = 0.75
    probs = model.predict_proba(X_test)
    y_pred = (probs[:,1] >= threshold)
    y_pred =  np.where(y_pred, 1, 0) 

    # Print accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print("\n\n Test accuracy:", accuracy, "\n\n")

    lgb.plot_metric(evals, metric='binary_logloss')
    plt.savefig(f'../models/lgbm_{timestamp}/binary_logloss_plot.png')

    lgb.plot_metric(evals, metric='auc')
    plt.savefig(f'../models/lgbm_{timestamp}/auc_plot.png')

    class_names = ["Nein", "Ja"]
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, cmap='Blues', colorbar=False)
    plt.savefig(f'../models/lgbm_{timestamp}/confusion_matrix.png')

    return y_pred, probs, accuracy

# %%
def main(crossvalidation: bool):
    if crossvalidation == False:
        # Split dataset
        folder_path = "../data/labeled_data/"
        X_train, y_train, X_val, y_val, X_test, y_test, features, weight_factor, timestamp, vocab = load_prepare_dataset(folder_path, only_text=False, test_size=lgbm_params["test_size"], augmentation=True, kfold=False)

        store_model = False
        show_preds = True

        gbm, evals = train_model(X_train, y_train, X_val, y_val, weight_factor)
        y_pred, probs, test_acc = evaluate_model(gbm, X_test, y_test, evals, timestamp)

        if show_preds:
            store_predictions(gbm, X_test, y_test, y_pred, probs, features, timestamp)

        if store_model:
            store_trained_model(gbm, test_acc, timestamp)

        plot_importance(gbm, max_num_features=10)
    else:
        # Split dataset
        folder_path = "../data/labeled_data/"
        store_model = False
        show_preds = False

        X_train, y_train, X_test, y_test, features, weight_factor, timestamp, vocab = load_prepare_dataset(folder_path, augmentation=True, test_size=lgbm_params["test_size"], kfold=True)

        kfold = KFold(n_splits=7, shuffle=True, random_state=42)
        evals_list = []

        for train_index, val_index in kfold.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            gbm, evals = train_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold, weight_factor)
            evals_list.append(evals)

            y_pred, test_acc = evaluate_model(gbm, X_test, y_test, evals, timestamp)

        if show_preds:
            store_predictions(gbm, X_test, y_test, y_pred, features, timestamp)

        if store_model:
            store_trained_model(gbm, test_acc, timestamp) 


# %%
if __name__ == "__main__":
    
    main(crossvalidation=False)



# %%
