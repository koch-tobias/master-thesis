# Component Identification for Geometric Measurements in the Vehicle Development Process Using Machine Learning

## 📖 Quick Index

* [💻 Installation](#-installation)
* [🐍 Usage](#-usage-in-python)
* [🚀 Updates](#-updates)

## 💻 Installation
1. Clone or download the repository
2. (optional) Create a virtuel environment: 
```bash
conda create -n envMesstool python=3.10
```
3. (optional) Activate the created environment 
```bash 
conda activate envMesstool 
```
4. Go to the root directory
5. Installation of the requirements: 
```bash 
pip install -r requirements.txt
``` 
6. Add the data folder to the root directory
7. Perfect, the messtool is ready to use! If not, run 
```bash 
pip install -e .
``` 
or 
```bash
python setup.py install
``` 

## 🐍 Usage in Python
The project contains 5 different pipelines. 
- Labeling pipeline: Preparation of raw data, feature selection, add label columns, and prelabeling with the trained models
- Data pipeline: Combine all datasets to one, feature engineering, data preprocessing, data augmentation, and splitting the dataset into train, validation, and testset
- Training pipeline: Training of the binary and/or multiclass models using grid search hyperparametertuning and k-fold-crossvalidation 
- Explainability pipeline: Create shap plots, tree plots, and store the feature importantance for the final models 
- Deployment pipeline: Deploy the model using FastAPI and Docker or using a Streamlit website

### Labeling pipeline
![Labeling pipeline](images/pipelines/labeling_pipeline.svg)

This pipeline is used for the preparation and prelabeling of the raw data. 
The following folder structure is required for this: </br>
master-thesis/ </br>
├─ data/ </br>
│  ├─ labeled/ </br>
│  ├─ pre_labeled/ </br>
│  ├─ processed/ </br>
│  ├─ preprocessed/ </br>
│  ├─ raw/ </br> 
│  ├─ raw_for_labeling/ </br>

The first step in using this pipeline is to add all raw datasets (excel files), which you want to add for training new models, to the raw_for_labeling folder. Each excel file contains the structure tree of one vehicle and have to include at least all relevant attributes, which are:
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

After that, run the **label.py** file which you can find here: </br>
master-thesis/  </br>
├─ src/ </br>
│  ├─ labeling_pipeline/ </br>
│  │  ├─ ___init___.py </br>
│  │  ├─ label.py </br>

This file will apply the following steps:
- **Data Preparation**: Keeps only vehicle parts from relevant modules. All parent folders are removed
- **Feature Selection**: Keeps only the features which are used for training the models
- **Add label columns**: Adding and initializing the label columns ("Relevant fuer Messung" with 0 and "Einheitsname" with "Dummy")
- **Prelabeling**: Using the trained model to identify the relevant car parts and classify a uniform name ("Einheitsname") for each relevant car part

After these steps, the prepared and prelabeled datasets are stored in the folder "pre_labeled". 
Now please check carefully if the records are labeled correctly. If not, you will have to correct this manually.
This is a critical part, because incorrectly labeled data can lead to a significant drop in performance!

After reviewing the pre-labeled datasets, move them to the labeled folder.

### Data pipeline
### Training pipeline
### Explainability pipeline
### Deployment pipeline

## 🚀 Updates

**2023.05.01**