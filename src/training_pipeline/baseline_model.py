from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, log_loss
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

import os
import time
from pathlib import Path
import yaml
from yaml.loader import SafeLoader
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from matplotlib.colors import LinearSegmentedColormap


import csv
from sklearn.model_selection import ParameterGrid

import sys
sys.path.append(os.getcwd())

from src.utils import store_trained_model, load_training_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def train_best_model(best_params):
    # Create a random forest classifier with the best parameters
    rf_classifier = RandomForestClassifier(criterion='log_loss',
                                        class_weight='balanced',
                                        verbose=2,
                                        random_state=42,
                                        **best_params,
                                        warm_start=True)

    # Train the model for multiple iterations while saving the loss value
    train_loss_values = []
    val_loss_values = []
    train_auc_values = []
    val_auc_values = []
    for n_estimators in range(10, best_params['n_estimators'], 10):
        rf_classifier.set_params(n_estimators=n_estimators)
        rf_classifier.fit(X_train, y_train)

        val_auc = roc_auc_score(y_val, rf_classifier.predict_proba(X_val)[:,1], average='macro')
        train_auc = roc_auc_score(y_train, rf_classifier.predict_proba(X_train)[:,1], average='macro')
        train_loss = log_loss(y_train, rf_classifier.predict_proba(X_train)[:,1])
        val_loss = log_loss(y_val, rf_classifier.predict_proba(X_val)[:,1])

        train_auc_values.append(train_auc)
        val_auc_values.append(val_auc)
        train_loss_values.append(train_loss)
        val_loss_values.append(val_loss)

    # Plot the auc values over the number of iterations
    plt.plot(range(10, best_params['n_estimators'], 10), train_auc_values, label='Train AUC')
    plt.plot(range(10, best_params['n_estimators'], 10), val_auc_values, label='Validation AUC')
    plt.xlabel('Number of Iterations')
    plt.ylabel('AUC')
    plt.title('Training and Validation AUC')
    plt.legend()
    plt.ylim([0, 1.2])
    plt.savefig('src/training_pipeline/trained_models/Random_forest/auc_plot.png')
    plt.close()

   # Plot the loss values over the number of iterations
    plt.plot(range(10, best_params['n_estimators'], 10), train_loss_values, label='Train Loss')
    plt.plot(range(10, best_params['n_estimators'], 10), val_loss_values, label='Validation Loss')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.ylim([0, 1.2])
    plt.savefig('src/training_pipeline/trained_models/Random_forest/loss_plot.png')
    plt.close()


    # Make predictions on the validation set
    val_predictions = rf_classifier.predict(X_val)

    # Validation confusion matrix
    val_cm = confusion_matrix(y_val, val_predictions)

    colors = ['#F4FFFF', '#BBFFFF', '#00FFFF', '#00E6E6', '#00BFBF', '#009999', '#007F7F'] 
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    fontsize = 18

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(20, 15))
    sns.heatmap(val_cm, annot=True, fmt="d", cmap=cmap, annot_kws={"size": fontsize})
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig('src/training_pipeline/trained_models/Random_forest/val_confusion_matrix.png')
    plt.close()

    # Make predictions on the test set
    test_predictions = rf_classifier.predict(X_test)

    # Validation confusion matrix
    test_cm = confusion_matrix(y_test, test_predictions)

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(test_cm, annot=True, fmt="d", cmap=cmap, annot_kws={"size": fontsize})
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig('src/training_pipeline/trained_models/Random_forest/test_confusion_matrix.png')
    plt.close()

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 300, 600],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create a random forest classifier
    rf_classifier = RandomForestClassifier(criterion='log_loss', class_weight='balanced', verbose=2, random_state=42)

    # Generate all possible combinations of parameters
    param_combinations = ParameterGrid(param_grid)

    # Store the results in a list
    results = []

    # Iterate over all parameter combinations
    for params in param_combinations:
        # Set the parameters for the random forest classifier
        rf_classifier.set_params(**params)

        # Fit the model on the training data
        start = time.time()
        rf_classifier.fit(X_train, y_train)
        stop = time.time()
        training_time = int(stop - start)

        # Make predictions on the validation set
        val_predictions = rf_classifier.predict(X_val)

        # Evaluate the model's sensitivity on the validation set
        val_recall = recall_score(y_val, val_predictions, average='binary', pos_label='Ja')

        # Evaluate the model's f1-score on the validation set
        val_f1 = f1_score(y_val, val_predictions, pos_label='Ja')

        # Evaluate the model's auc on the validation set
        val_auc = roc_auc_score(y_val, rf_classifier.predict_proba(X_val)[:,1], average='macro')

        # Make predictions on the test set
        test_predictions = rf_classifier.predict(X_test)

        # Evaluate the model's sensitivity on the validation set
        test_recall = recall_score(y_test, test_predictions, average='binary', pos_label='Ja')

        # Evaluate the model's f1-score on the validation set
        test_f1 = f1_score(y_test, test_predictions, pos_label='Ja')

        # Evaluate the model's auc on the validation set
        test_auc = roc_auc_score(y_test, rf_classifier.predict_proba(X_test)[:,1], average='macro')

        # Store the parameters and results in the list
        results.append((params, val_auc, val_recall, val_f1, test_auc, test_recall, test_f1, training_time))

    # Create a DataFrame from the results list
    df_results = pd.DataFrame(results, columns=['Parameters', 'Validation AUC', 'Validation Sensitivity', 'Validation F1_Score', 'Test AUC', 'Test Sensitivity', 'Test F1_Score', 'Training Time'])

    best_params = df_results.loc[df_results['Validation AUC'].idxmax()]['Parameters']

    train_best_model(best_params)
    # Save the DataFrame to an Excel file
    df_results.to_excel('src/training_pipeline/trained_models/Random_forest/rf_cv_results.xlsx', index=False)

    return results

data_folder = Path(config["train_settings"]["folder_processed_dataset"])
X_train, y_train, X_val, y_val, X_test, y_test, df_preprocessed, df_train, df_val, df_test, weight_factor = load_training_data(data_folder, binary_model=True)

# Select the categorical feature column
categorical_feature = df_preprocessed['Benennung (dt)']

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the categorical feature
encoded_feature = label_encoder.fit_transform(categorical_feature)

# Replace the original categorical feature column with the encoded values
df_preprocessed['CategoricalFeature'] = encoded_feature

X_train, X_val, y_train, y_val = train_test_split(df_preprocessed[['CategoricalFeature','center_x', 'center_y', 'center_x', 'theta_x', 'theta_y', 'theta_z', 'volume', 'Wert']], df_preprocessed['Relevant fuer Messung'], test_size=0.7, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

'''
# Fit and transform the column
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
encoded_column = one_hot_encoder.fit_transform(df_preprocessed[['Benennung (dt)']]).toarray()

# Create a new DataFrame with the encoded column
encoded_df = pd.DataFrame(encoded_column, columns=one_hot_encoder.get_feature_names_out(['Benennung (dt)']))

df = df_preprocessed[['center_x', 'center_y', 'center_x', 'theta_x', 'theta_y', 'theta_z', 'volume', 'Wert']]
df_label = df_preprocessed[['Relevant fuer Messung']]
# Concatenate the encoded DataFrame with the remaining features
df_preprocessed = pd.concat([df, encoded_df], axis=1)

X_train, X_val, y_train, y_val = train_test_split(df_preprocessed, df_label['Relevant fuer Messung'], test_size=0.7, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
'''

best_model = train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)