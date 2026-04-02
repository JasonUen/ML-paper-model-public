# ML-paper-model
This repository contains scripts to conduct analyses in our publication :
**Geospatial analytics and machine learning for forecasting county-level food waste in U.S. retail markets**  
*Tinn-Shuan Uen, Luis F. Rodríguez*  
Resources, Conservation & Recycling, 2026.  
https://doi.org/10.1016/j.resconrec.2026.108906

## Project structure
1. Pre-modeling analysis
2. Building, tuning, and evaluating regression models.

---

## Features
- Preprocessing scripts for data transformation (Box-Cox, normalization)
- Multiple ML algorithms with hyperparameter tuning
- Deep Neural Network implementation (Keras/TensorFlow)
- Utility functions in `utils.py` for:
  - Data loading

---

## Python version
This project requires: Python 3.10.x for TensorFlow compatibility.

A virtual environment is recommended:

```bash
python3 -m venv ml_venv
source ml_venv/bin/activate   # macOS/Linux
ml_venv\Scripts\activate      # Windows
```

## Installation
1. clone the repo
```bash
git clone https://github.com/JasonUen/ML-paper-model.git
cd ML-paper-model
```
2. Install the package in the editable mode
```bash
pip install -r requirements.txt
```

## Data availability
The processed and raw datasets used in this project are not included in the repository for privacy and size considerations.
Data can be provided upon request.

## Models included
	•	Multiple Linear Regression (MLR)
	•	Lasso & Ridge Regression
	•	Random Forest Regression (RF)
	•	Gradient Boosting Regression (GBR)
 	•	Adaptive Boosting Regression (ADA)
	•	Support Vector Regression (SVR)
	•	Deep Neural Network (Keras)

## Post-analysis
  - Unified plotting of model performance (MAE, RMSE, R²)  
  - Feature importance: tree-based FI, permutation FI, and coefficients where available.

---

## Steps in this project

### 1. Request data
The raw datasets required for running this project are **not included in this repository** for simplicity and size reasons.  
Please contact the authors to request access.

### 2. Configure data path
Once you have the data, update the `path_base_data` in **`ml_scripts/ML_Pre.py`** (around line 35):

### 3. Run data preprocessing
Run ML_Pre.py to generate processed datasets and pre-modeling analysis results.
This will create a folder named data_analysis/ in your project root directory, containing: clean, transformed data, correlation heatmaps, and statistics summary

### 4. Evaluate models
Each algorithm has a dedicated eval_*.py file under ml_scripts/. Example: eval_RF.py

### 5. Compare models
Use results_analysis.py in the results folder to compare model performance (RMSE, MAE, R²).
Use FI_plots.py in the ml_scripts folder to visualize the results.


---

## Authors
- Tinn-Shuan (Jason) Uen
- Luis F. Rodriguez
