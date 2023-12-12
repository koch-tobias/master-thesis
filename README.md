# Component Identification for Geometric Measurements in the Vehicle Development Process Using Machine Learning

## 🚘🔍 CaPI (Car Part Identifier)
CaPI is an AI system developed as part of my master's thesis and is based on two machine learning models. CaPI identifies car parts that are relevant for geometric measurements during the virtual vehicle development process. CaPI is one of three components of a measurement tool that automates the process of measuring various dimensions and comparing them to guidelines and targets. </br>
The input for the inference models is an Excel file that contains the structural list of a vehicle model, including components and their metadata. For example:
![Sample excerpt input](images/sample_input.png)

The first goal of the AI system (model 1) is to classify the components of a vehicle as relevant and not relevant for the geometric measurements (binary classification task). The second goal (model 2) is to generate uniformly coded designations for the relevant car components (multi-class classification task).

The output of the AI system is a list of all relevant components found in the structure list, with the appropriate, uniformly coded name for each component.
This output is used by a CATIA macro to load the car parts into a CATIA parametric model, which then automatically performs the measurements and the comparisons with the guidelines. 

CaPI is accessible via a REST API or a website. The REST-API is implemented for production to integrate the AI system in the measurement tool and the website is set up for the development process to test models and to get quick feedback from the users.

A detailed description of the project can be found in the master's thesis.

## 📖 Quick Index

* [💻 Installation](#-installation)
* [⛏️ Architecture](#-architecture)
* [🐍 Usage](#-usage-in-python)
* [🚀 Updates](#-updates)

## 💻 Installation
1. Clone or download the repository
2. Create a virtual environment
```bash
conda create -n envMesstool python=3.10
```
3. Activate the created environment 
```bash 
conda activate envMesstool 
```
4. Go to the root directory of the project
5. Install the requirements
```bash 
pip install -r requirements.txt
``` 
6. DONE, the code is ready to use! 

## 🐍 Usage in Python
Since there is no data provided in this repository, the first step is to **add the data to the root folder** by using the following folder structure:</br>
master-thesis/ </br>
├─ data/ </br>
│  ├─ labeled/ </br>
│  ├─ pre_labeled/ </br>
│  ├─ processed/ </br>
│  ├─ raw/ </br> 
│  ├─ raw_for_labeling/ </br>

### Prelabeling New Data
For pre-labeling new data, the first step is to **add the raw data sets** (Excel files) to the folder "data/raw_for_labeling". 
Each Excel file must contain the structural list of a vehicle model and needs at least the following attributes:
- Sachnummer
- Benennung (dt)
- X-Min
- X-Max
- Y-Min
- Y-Max
- Z-Min
- Z-Max
- Wert
- Einheit
- Gewichtsart
- Kurzname
- L-Kz.
- L/R-Kz.
- Modul (Nr)
- Code
- ox
- oy
- oz
- xx
- xy
- xz
- yx
- yy
- yz
- zx
- zy
- zz

After adding all files, you can specify the following settings in the **src/config.yaml** file: 
- **binary_column** and **multiclass_column**: Names of the label columns
- **binary_label_1** and **binary_label_0**: The labels for the binary classification
- **keep_modules**: Modules that should be kept
- **relevant_features**: Features that should be kept

Now, run the following command from the root directory: </br>
```bash
python src\deployment\data_prelabeling.py 
``` 

This file executes the following steps:
- **Add Label Columns**: Add and initialize the label columns. ("Relevant fuer Messung" with "Nein" and "Einheitsname" with "Dummy")
- **Prelabeling**: Uses the trained models to identify the relevant car components and classify a uniform name ("Einheitsname") for each relevant component.

After these steps, the prelabeled datasets are stored in the folder **"data/pre_labeled"**. 
**Now, please check if the samples are labeled correctly.** If not, please correct the labels manually.
This is a critical part because incorrectly labeled data can lead to a significant drop in the model performance.
As an assistance to correctly label the datasets and to quickly detect possible errors, a checklist (excel) with all vehicles and the determined relevant components is available in the data folder.

After reviewing the pre-labeled data sets, move the labeled datasets to the **"data/labeled"** folder and move the raw datasets from "data/raw_for_labeling" to the folder "data/raw" (add the derivat to the file name).

### Generate the Training Data

Before running this process, you can specify the following settings in the **"src/config.yaml"** file:
- **seed**: Seed for the train, val, test split
- **cut_percent_of_front**: All components located in the front x percent are removed. (For example, cut_percent_of_front=0.18 means that all car parts up to the windshield will be removed. cut_percent_of_front=0 means that no car parts will be removed)
- **car_part_designation**: Specify the column that contains the car part designation as text
- **use_only_text**: If true, only the car part designation will be used as a feature. All other features are not considered.
- **normalize_numerical_features**:  If true, numerical features will be normalized.
- **bounding_box_features_original**: List of all features which represent the bounding boxes
- **features_for_model**: List of features which will be used, in addition to the car part designation, to train the model
- **train_val_split**: Split into (1-x)\*100 % training and x\*100 % validation set. (x=[0,1])
- **val_test_split**: Split into (1-x)\*100 % validationset and x\*100 % testset. (x=[0,1])

Now, to generate a new training, validation, and test split for training the models, run the data preprocessing pipeline from the root directory using the command: </br>
```bash
python src\data_preprocessing_pipeline\data_preprocessing.py
``` 

This file executes the following steps:
- **Combine Datasets**: All datasets which are in the data folder "data/labeled" will be combined into one dataframe.
- **Feature Engineering**: The bounding box information is transformed to reduce the number of features. In the original data set, a bounding box is represented by the minimum and maximum values in x-, y-, and z-direction, a shift vector, and a rotation matrix.
Here, this information is used to transform them into length, width, height, the center point in x-, y-, and z-direction, and an orientation vector of the bounding box. This reduces the number of features for representing the bounding box from 18 to 9. In addition, two more features (volume and density) are calculated.
- **Data Cleaning**: This involves a preselection of irrelevant car components based on the bounding box features (volume and position). In addition, component designations are cleaned by removing punctuation and frequently occurring words without information. The text data is then converted to numeric vectors.
- **Data Augmentation**: The synthetic designations are generated by adding random mistakes, switching words, or generating new names using GPT3.5. The synthetic bounding box information is randomly generated. However, it must be within a validated range in terms of position, length, width, height, and volume in reference to the original components of the same class.
- **Training, Validation, and Test Split**: To ensure that the datasets are balanced across classes a stratified training, validation, and test split is performed with the additional help of data augmentation techniques to create synthetic car parts. </br> 
 
The outputs of this process are then stored in a newly created folder with the naming convention "YYYYMMDD_Time" inside the folder **"data/processed"**:
- binary folder:
  - Dict with train, validation, and test split in dataframes (pickle file)
  - Dict with train, validation, and test split in numpy arrays (pickle file)
  - Train, validation, and test split distribution plot
  - Class distribution plot
- multiclass folder:
  - Dict with train, validation, and test split in dataframes (pickle file)
  - Dict with train, validation, and test split in numpy arrays (pickle file)
  - Train, validation, and test split distribution plot
  - Class distribution plot
- feature_distribution folder: Contains a distribution plot for each numerical feature
- boundingbox_features.pkl: Stores all features used additionally to the car part designation
- label_encoder.pkl: Label encoder for inverse encoding
- processed_data.csv: The combined dataframe used to create the train, validation, and test split
- vectorizer.pkl: Used in production to convert the text data in the same way as the model was trained
- vocabulary.pkl: Stores the used vocabulary

### Training
After creating the data set, you can train and evaluate new models. Currently, the gradient boosting methods LightGBM, XGBoost, and CatBoost and the deep learning method PyTorch Tabular are available. </br>
Before running this process, you can specify the following settings in the **"src/config.yaml"** file:
- **folder_processed_dataset**: Path to the dataset folder
- **train_binary_model**: True = Training a binary model
- **train_multiclass_model**: True = Training a multiclass model
- **ml-method**: Choose between lgbm, catboost, or xgboost
- **k-folds**: Number of folds used for cross-validation
- **early_stopping**: Patience which is used for early stopping
- **top_x_models_for_cv**: x\*100 percent of the trained models with grid search hyperparameter tuning will be used for validation with k-fold cross-validation
- **n_estimators**: Number of iterations the model will be trained
- **Specific method setting**: Metrics, boosting type, and hyperparameters for each machine learning method

If train_binary_model and train_multiclass_model are both declared true, the process executes first the training of the binary model and then the training of the multi-class model, but both are stored in the same main folder.

After setting the desired training parameters, the training process can be started by executing the **train.py** file from the root directory by using the command: </br>
```bash
python src\training_pipeline\train.py 
``` 

This file executes the following steps:
- **Hyperparameter Tuning**: The first step after loading the datasets is the tuning of the hyperparameters via grid search. Here, 81 models are trained iteratively by varying over 4 hyperparameters. 
- **Cross-Validation**: After grid search, the top x % [default = 5] of the models are selected by the highest F2-score on the validation set and then validated using k-fold cross-validation.
- **Train the Final Model**: The model with the highest F2-score after cross-validation on the validation set is then selected as the "best" model. Then, the hyperparameters of this model are used to train the final model on a larger train set that combines the previous train set and the test set to use the entire available data for training. The validation of the final model is still performed on the validation set as before. 

The trained models and their files to validate and compare the models are stored in the following folder: </br>
master-thesis/ </br>
├─ src/ </br>
│  ├─ training_pipeline/ </br>
│  │  ├─ trained_models/ </br>

To check how the trained models have performed, the following files are stored after training:
- For the best model after cross-validation and the final model
  - F2-score plot
  - Loss plot 
  - Confusion matrix on the test set 
- Table (csv-file) with all results for hyperparameter tuning and cross-validation
- Table (csv-file) with all wrongly classified car par parts 
- Logging file (txt) with the hyperparameters the stored models are trained on, and the dataset settings

If the newly trained models have a better performance, update the final models in the following folder with the newly trained models: (The existing models should be moved to the "final_models/Archiv" folder) </br>
master-thesis/ </br>
├─ final_models/ </br>

**The models in the "final_models" folder are used for inference.**

**Explainability**

For each of the final models, insights about the predictions can be generated by executing: </br>
```bash
python python src\explainability\xAi.py 
``` 

It uses the models in the folder "final_models" to generate insights in the following way:
- **Feature Importance**: Creates an Excel file with all the features that the model has trained on and their importance to the model output. 
- **SHAP**: Creates a beeswarm plot, which also helps to understand, which features and which feature values affect the models' prediction.
- **Decision Trees**: Plot a decision tree based on a selected tree index.

The results of this pipeline are stored in the folder "final_models" for each classification task (binary and multiclass), for example: </br>
master-thesis/ </br>
├─ final_models/ </br>
│  ├─ Binary_model/ </br>
│  │  ├─ xAi/ </br>

### Deployment

The current options to deploy the models are a streamlit website, and a REST-API, which are both hosted on an AWS server. </br>
The streamlit website is used for testing, debugging, and getting feedback from the department/users and can be addressed by the following URL:
```
http://10.3.13.137:7071/
```
If you want to run the website locally, run the following command from the root folder:
```bash
streamlit run src\deployment\website.py
```

The REST-API developed with FastAPI and virtualized with docker is developed for production and can be addressed by sending a request to the following URL: </br>
```
http://10.3.13.137:7070/api/get_relevant_parts/
```

The input is an Excel file of the structural of a selected vehicle with all components and their metadata. The output is a JSON file, which contains a list of the identified relevant components, as follows:
```
{
  "1234567": [
    "RESTDACH",
    "DACHAUSSENHAUT"
  ],
  "23456789": [
    "AUSSENHAUT HKL OBEN",
    "HECKKLAPPE AUSSENHAUT"
  ],
  "98765432": [
    "ZB SKYROOF DACHMODUL",
    "GLASDACH (SA)"
  ]
}
```
You can test the API using the following python script (src/deployment/api_request.py), which sends a Excel file to the API and returns the identified car parts:
```python
import requests

import yaml
from yaml.loader import SafeLoader
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

url = "http://10.3.13.137:7070/api/get_relevant_parts/" 

file_path = config["paths"]["test_file_path"]
files = {"file": open(file_path, "rb")}
headers = {"accept": "application/json"}

proxies = {
  "http": None,
  "https": None
  }

response = requests.post(url, files=files, headers=headers, proxies=proxies)
print(response.content)
```

For running the API locally, run the following command from the root directory:
```bash
uvicorn src.deployment.api:app --reload --port 5000
```

Now you can access the local API using:
```bash
http://localhost:5000/docs
 ```

If you want to quickly test models locally, add the path of the testing file to the parameter **test_file_path** in the "src/config.yaml" file and run the following command from the root directory: </br>
```bash
python src\deployment\classification.py 
``` 
It uses the models in the "final_models" folder and outputs the dataframe with all identified car parts and a list of all car parts which are not identified. The output is only visible in the terminal and is not stored.


### Unittests
In order to test functions individually and independently for proper operation, unit tests were developed using the pyTest library. These can be executed in the root folder by using the following command:
``` 
pytest 
``` 
