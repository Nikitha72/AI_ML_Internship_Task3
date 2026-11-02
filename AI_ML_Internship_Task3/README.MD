# Task 3: Build & Evaluate a Machine Learning Model

## Project
Student Pass/Fail Prediction (Classification).

## Dataset note
`student_performance_task3.csv` is synthetic practice data for learning. The internship brief says Task 3 should use the dataset prepared in Task 2. For final submission, replace this practice CSV with your Task 2 dataset and define a valid target column; document the data source.

## What the notebook covers
- Problem statement, features (X) and target (y)
- Train/test split
- Logistic Regression and Random Forest
- Predictions and classification report
- Accuracy, precision, recall and F1-score
- Confusion matrices and model comparison
- Example prediction and saving the best model

## Setup
Install packages:
```powershell
python -m pip install pandas numpy matplotlib scikit-learn joblib jupyter
```

Open `Task3_ML_Model_Development.ipynb` in VS Code, select your Python environment/kernel, and use Run All.

## Outputs generated when notebook runs
- `task3_model_comparison.csv`
- `task3_confusion_matrices.png`
- `task3_model_comparison.png`
- `task3_best_model.joblib`
