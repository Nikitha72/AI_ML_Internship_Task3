
# ============================================================
# TASK 3: BUILD & EVALUATE A MACHINE LEARNING MODEL
# Project: Student Pass/Fail Prediction
# ============================================================


# 1. IMPORT REQUIRED LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("Libraries imported successfully!")


# 2. SET PROJECT DIRECTORY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_DIR)

print("\nProject directory:", BASE_DIR)


# 3. CREATE OUTPUT FOLDER

OUTPUT_DIR = os.path.join(BASE_DIR, "task3_outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Output folder created successfully!")


# 4. LOAD DATASET

DATA_FILE = os.path.join(
    BASE_DIR,
    "student_performance_task3.csv"
)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Please place student_performance_task3.csv "
        "inside your Task 3 folder."
    )

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")
print("Dataset used:", os.path.basename(DATA_FILE))

print("\nFirst five records:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()


# 5. DATA CLEANING

df.columns = df.columns.str.strip()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:", df.duplicated().sum())

df = df.drop_duplicates().copy()


# 6. DEFINE INPUT FEATURES (X) AND TARGET (Y)

if "Pass_Fail" in df.columns:

    y = df["Pass_Fail"].astype(str).str.strip().str.title()

    excluded_columns = [
        "Pass_Fail",
        "Student_ID",
        "Final_Exam_Score"
    ]

    X = df.drop(
        columns=[
            col for col in excluded_columns
            if col in df.columns
        ]
    )

elif "G3" in df.columns:

    y = np.where(
        df["G3"] >= 10,
        "Pass",
        "Fail"
    )

    X = df.drop(columns=["G3"])

else:

    raise ValueError(
        "Dataset must contain Pass_Fail or G3 column."
    )


# Remove missing target records

valid_rows = pd.Series(
    y,
    index=X.index
).notna()

X = X.loc[valid_rows].copy()

y = pd.Series(
    y,
    index=X.index,
    name="Pass_Fail"
)

print("\nInput features:")
print(X.columns.tolist())

print("\nTarget variable distribution:")
print(y.value_counts())


# 7. TRAIN AND TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# 8. DATA PREPROCESSING

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()


# Numeric preprocessing

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical preprocessing

try:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )

except TypeError:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=True
    )


categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", encoder)
])


# Combine preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)

print("\nPreprocessing completed!")


# 9. IMPLEMENT MACHINE LEARNING MODELS

models = {

    "Logistic Regression": Pipeline(steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]),

    "Random Forest": Pipeline(steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])

}


# 10. TRAIN BOTH MODELS

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    print(name, "training completed!")


# 11. GENERATE PREDICTIONS

predictions = {}

for name, model in models.items():

    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    print("\nPredictions using:", name)

    print(y_pred[:10])


# 12. EVALUATE MODEL PERFORMANCE

results = []

for name, model in models.items():

    y_pred = predictions[name]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    results.append({

        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1

    })

    print("\n================================")
    print(name)
    print("================================")

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


# 13. MODEL COMPARISON

comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.sort_values(
    by="F1_Score",
    ascending=False
)

print("\nMODEL COMPARISON")

print(comparison_df)

comparison_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "task3_model_comparison.csv"
    ),
    index=False
)


# 14. CONFUSION MATRIX

class_labels = sorted(y.unique())

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

for ax, (name, y_pred) in zip(
    axes,
    predictions.items()
):

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=class_labels
    )

    display_matrix = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_labels
    )

    display_matrix.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    ax.set_title(name)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "task3_confusion_matrices.png"
    )
)

plt.show()


# 15. MODEL PERFORMANCE GRAPH

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score"
]

plot_data = comparison_df.set_index("Model")[metrics]

plot_data.plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1)
)

plt.title("Machine Learning Model Comparison")

plt.xlabel("Models")
plt.ylabel("Performance Score")

plt.xticks(rotation=0)

plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "task3_model_comparison.png"
    )
)

plt.show()


# 16. SELECT THE BEST MODEL

best_model_name = comparison_df.iloc[0]["Model"]

best_model = models[best_model_name]

print("\nBest Model:", best_model_name)

print(
    "Best F1 Score:",
    round(comparison_df.iloc[0]["F1_Score"], 4)
)


# 17. SAVE THE TRAINED MODEL

joblib.dump(
    best_model,
    os.path.join(
        OUTPUT_DIR,
        "task3_best_model.joblib"
    )
)

print("\nBest model saved successfully!")


# 18. SAVE ACTUAL VS PREDICTED RESULTS

best_predictions = best_model.predict(X_test)

prediction_results = pd.DataFrame({

    "Actual": y_test.to_numpy(),

    "Predicted": best_predictions

})

prediction_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "task3_test_predictions.csv"
    ),
    index=False
)

print("\nActual vs Predicted Results:")

print(prediction_results.head(10))


# 19. EXAMPLE STUDENT PREDICTION

sample_student = X_test.iloc[[0]]

sample_prediction = best_model.predict(
    sample_student
)

print("\nExample Student Details:")

print(sample_student)

print("\nPredicted Student Result:")

print(sample_prediction[0])

print("\nActual Student Result:")

print(y_test.iloc[0])


# 20. FINAL PROJECT SUMMARY

print("\n======================================")
print("TASK 3 COMPLETED SUCCESSFULLY")
print("======================================")

print("Dataset:", os.path.basename(DATA_FILE))

print("Total Records:", len(df))

print("Training Records:", len(X_train))

print("Testing Records:", len(X_test))

print("\nAlgorithms Used:")
print("1. Logistic Regression")
print("2. Random Forest")

print("\nBest Performing Model:")
print(best_model_name)

print("\nGenerated Output Files:")

for filename in os.listdir(OUTPUT_DIR):
    print("-", filename)

print("\nALL TASK 3 OPERATIONS COMPLETED!")
