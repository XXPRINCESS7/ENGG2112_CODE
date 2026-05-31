# Solar Power Generation Prediction Project

This repository contains the code for an ENGG2112 machine learning project that predicts solar power generation using weather and solar-related features.

The project is divided into two main parts:

1. **Model Development**
2. **MVP Application**

The model development part focuses on data preprocessing, model training, model comparison, and result visualisation. The MVP part provides a Streamlit dashboard that demonstrates the prediction workflow in an interactive way.

---

## Project Overview

The aim of this project is to predict solar power generation, represented by `generated_power_kw`, based on weather and solar-related input features.

Several regression models are developed and compared to identify which model performs best for the solar power prediction task. The final MVP application allows users to view model performance, select a model, choose a sample input, and compare the predicted solar power with the actual value.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── model/
│   ├── model.py
│   └── spg.csv
│
└── mvp/
    ├── app.py
    └── sample_data.csv
```

---

## 1. Model Development

The `model/` folder contains the full machine learning workflow used for model development and analysis.

### Main File

```text
model/model.py
```

### Main Functions

The model development script includes:

- Loading the solar power dataset
- Removing duplicate rows
- Converting angular features into sine and cosine components
- Splitting data into training and testing sets
- Applying feature standardisation where required
- Training multiple regression models
- Evaluating model performance
- Comparing models using numerical metrics
- Plotting actual-vs-predicted results
- Plotting residuals
- Analysing feature importance

### Models Used

The following machine learning models are compared:

- Linear Regression
- Random Forest Regressor
- Support Vector Regression
- Multi-Layer Perceptron Regressor
- XGBoost Regressor

### Evaluation Metrics

The models are evaluated using:

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **R² Score**: Coefficient of Determination

Lower MAE and RMSE values indicate smaller prediction errors. A higher R² score indicates that the model explains more variation in the target variable.

### Run the Model Script

To run the full model development script, use:

```bash
python model/model.py
```

The script expects the full dataset file `spg.csv` to be located inside the `model/` folder.

---

## 2. MVP Application

The `mvp/` folder contains the final Streamlit MVP application used for the project demonstration.

### Main File

```text
mvp/app.py
```

### Purpose

The MVP is designed as an interactive dashboard for demonstrating the solar power prediction workflow. It uses a built-in sample dataset, so no CSV upload is required during the presentation demo.

### MVP Features

The Streamlit MVP allows users to:

- View the built-in solar dataset summary
- Compare model performance using MAE, RMSE, and R²
- Select a machine learning model
- Choose one prepared test sample
- View predicted solar power
- Compare predicted power with the actual power value
- View the absolute prediction error
- Inspect actual-vs-predicted plots
- Inspect residual plots
- View feature importance where available

### Run the MVP

To run the Streamlit MVP application, use:

```bash
python -m streamlit run app.py
```

The MVP automatically uses:

```text
mvp/sample_data.csv
```

No manual CSV upload is required.

---



## Requirements

This project uses the following main Python libraries:

- streamlit
- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost

---

## How to Use This Project

### Option 1: Run the Model Development Code

Use this option if you want to reproduce the full machine learning workflow.

```bash
python model/model.py
```

This will run the preprocessing, model training, evaluation, and visualisation process.

### Option 2: Run the MVP Dashboard

Use this option if you want to open the final interactive demo.

```bash
python -m streamlit run mvp/app.py
```

This will open the Streamlit application in a browser.

---

## Project Workflow

The overall workflow of this project is:

1. Load the solar power dataset.
2. Remove duplicate records.
3. Transform angular features into sine and cosine components.
4. Split the dataset into training and testing sets.
5. Standardise features for scale-sensitive models.
6. Train multiple regression models.
7. Evaluate model performance using MAE, RMSE, and R².
8. Compare model performance.
9. Visualise prediction results and residuals.
10. Build a Streamlit MVP to demonstrate the prediction process.

---

## Notes

The MVP uses a smaller built-in sample dataset for demonstration purposes. The full model development script may use a larger dataset for training and analysis.

The prediction results are based on historical data patterns. Therefore, the model should be treated as a decision-support and demonstration tool rather than a guaranteed forecasting system. Its performance may vary when applied to new locations, unseen weather conditions, different sensors, or different datasets.

---

## Project Components Summary

| Component | Folder | Main File | Purpose |
|---|---|---|---|
| Model Development | `model/` | `model.py` | Full preprocessing, training, evaluation, and visualisation |
| MVP Application | `mvp/` | `app.py` | Interactive Streamlit dashboard for demonstration |
| Dependencies | Root folder | `requirements.txt` | Python package requirements |
| Documentation | Root folder | `README.md` | Project explanation and running instructions |

---

## Author

ENGG2112 Solar Power Generation Prediction Project
