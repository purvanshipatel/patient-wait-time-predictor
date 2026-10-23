
# ============================================================
# PROJECT: PATIENT QUEUE WAIT TIME PREDICTOR
# PHASE 1: DATA PREPARATION AND MACHINE LEARNING
# ============================================================

# 1. IMPORT LIBRARIES

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

FILE_NAME = "ed_patient_data.csv"

if not os.path.exists(FILE_NAME):
    raise FileNotFoundError(
        "Dataset not found. Put ed_patient_data.csv "
        "inside the Spyder working directory."
    )

df = pd.read_csv(FILE_NAME)

# Clean column names
df.columns = df.columns.astype(str).str.strip()

print("\n========== DATASET LOADED ==========")
print("Dataset shape:", df.shape)

print("\nAll column names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Remove billed amount column
df = df.drop(columns=["billed_amount_usd"], errors="ignore")

print("Billed amount column removed.")
print(df.columns.tolist())



# ============================================================
# 3. SELECT WAITING-TIME TARGET COLUMN
# ============================================================

TARGET_COLUMN = "wait_time_doctor_min"

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' was not found. "
        f"Available columns: {df.columns.tolist()}"
    )

print("\nSelected target column:", TARGET_COLUMN)



# ============================================================
# 4. CLEAN DATASET
# ============================================================

# Remove duplicate rows
df = df.drop_duplicates().copy()

# Find arrival_datetime without using normalized_columns
arrival_column = None

for col in df.columns:
    if col.strip().lower().replace("_", "") == "arrivaldatetime":
        arrival_column = col
        break

# Convert arrival time to datetime and extract features
if arrival_column is not None:

    df[arrival_column] = pd.to_datetime(
        df[arrival_column],
        errors="coerce"
    )

    df["arrival_hour"] = df[arrival_column].dt.hour
    df["arrival_dayofweek"] = df[arrival_column].dt.dayofweek
    df["arrival_month"] = df[arrival_column].dt.month

# Convert target column to numeric
TARGET_COLUMN = "wait_time_doctor_min"

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN],
    errors="coerce"
)

# Remove missing target values
df = df.dropna(subset=[TARGET_COLUMN]).copy()

# Remove negative waiting times
df = df[df[TARGET_COLUMN] >= 0].copy()

print("\n========== SECTION 4 COMPLETED ==========")
print("Arrival column:", arrival_column)
print("Target column:", TARGET_COLUMN)
print("Cleaned dataset shape:", df.shape)
print("\nRemaining missing target values:")
print(df[TARGET_COLUMN].isnull().sum())



# ============================================================
# 5. PREPARE FEATURES (X) AND TARGET (y)
# ============================================================

# Select the target column
TARGET_COLUMN = "wait_time_doctor_min"

# Check whether the target exists
if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

# Convert target values to numeric
df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN],
    errors="coerce"
)

# Remove rows with missing or invalid target values
df = df.dropna(subset=[TARGET_COLUMN]).copy()

# Remove negative waiting times
df = df[df[TARGET_COLUMN] >= 0].copy()

# Separate target BEFORE preparing the input features
y = df[TARGET_COLUMN].copy()

# Columns that must not be used as input features
exclude_columns = [
    TARGET_COLUMN,          # Target itself
    "patient_id",           # Patient identifier
    "visit_id",             # Visit identifier
    "id",                   # Generic identifier
    "arrival_datetime",     # Raw datetime
    "wait_time_triage_min"  # Excluded to avoid timing leakage
]

# Drop only columns that actually exist
columns_to_drop = [
    col for col in exclude_columns
    if col in df.columns
]

X = df.drop(columns=columns_to_drop).copy()

# Remove columns containing only missing values
X = X.dropna(axis=1, how="all")

# Remove constant columns
X = X.loc[:, X.nunique(dropna=False) > 1]

# Keep only rows with finite target values
valid_rows = np.isfinite(y.to_numpy())

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

# Validate the prepared data
if len(X) < 10:
    raise ValueError(
        "Not enough valid rows to train the models."
    )

if X.shape[1] == 0:
    raise ValueError(
        "No usable input features remain."
    )

# Display the final prepared data
print("\n========== SECTION 5 COMPLETED ==========")

print("Target column:", TARGET_COLUMN)

print("\nInput features:")
print(X.columns.tolist())

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nFirst five target values:")
print(y.head())

print("\nMissing target values:", y.isnull().sum())
print("Negative target values:", (y < 0).sum())


# ============================================================
# 6. PREPARE NUMERIC AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number", "bool"]
).columns.tolist()

transformers = []

if numeric_features:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    categorical_pipeline = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    transformers.append(
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)


# ============================================================
# 7. SPLIT DATA INTO TRAINING AND TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# 8. CREATE THREE REGRESSION MODELS
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 9. TRAIN AND EVALUATE MODELS
# ============================================================

results = []
trained_models = {}
test_predictions = {}

for model_name, model in models.items():

    print("\nTraining:", model_name)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test, predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[model_name] = pipeline
    test_predictions[model_name] = predictions

    print("MAE:", round(mae, 2), "minutes")
    print("RMSE:", round(rmse, 2), "minutes")
    print("R2 Score:", round(r2, 3))


# ============================================================
# 10. COMPARE MODELS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE",
    ascending=True
).reset_index(drop=True)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# Select the model with the lowest test MAE
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
best_predictions = test_predictions[best_model_name]

print("\nSelected model:", best_model_name)


# ============================================================
# 11. GRAPH: MODEL PERFORMANCE COMPARISON
# ============================================================

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="MAE"
)

plt.title("Model Comparison - Mean Absolute Error")
plt.xlabel("Regression Model")
plt.ylabel("MAE (minutes)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# ============================================================
# 12. GRAPH: ACTUAL VS PREDICTED WAITING TIME
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.6
)

min_value = min(
    float(y_test.min()),
    float(np.min(best_predictions))
)

max_value = max(
    float(y_test.max()),
    float(np.max(best_predictions))
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title(
    "Actual vs Predicted Waiting Time - "
    + best_model_name
)

plt.xlabel("Actual Waiting Time (minutes)")
plt.ylabel("Predicted Waiting Time (minutes)")
plt.tight_layout()
plt.show()


# ============================================================
# 13. SAVE MODEL, METADATA AND RESULTS
# ============================================================

joblib.dump(
    best_model,
    "best_wait_time_model.pkl"
)

joblib.dump(
    {
        "feature_columns": X.columns.tolist(),
        "target_column": TARGET_COLUMN,
        "model_name": best_model_name
    },
    "project_metadata.pkl"
)

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\n========== PHASE 1 COMPLETED ==========")
print("Best model:", best_model_name)
print("Target column:", TARGET_COLUMN)
print("Saved model: best_wait_time_model.pkl")
print("Saved metadata: project_metadata.pkl")
print("Saved results: model_comparison.csv")



# ============================================================
# PHASE 2 - STEP 1: VERIFY SAVED MODEL
# ============================================================

import os
import joblib
import numpy as np

print("\n========== PHASE 2: MODEL VERIFICATION ==========")

# 1. Check saved files

required_files = [
    "best_wait_time_model.pkl",
    "project_metadata.pkl",
    "model_comparison.csv"
]

for file_name in required_files:
    if os.path.exists(file_name):
        print(file_name, "- FOUND")
    else:
        print(file_name, "- NOT FOUND")

# 2. Load the saved model and metadata

if not all(os.path.exists(f) for f in required_files):
    raise FileNotFoundError(
        "One or more project files are missing. "
        "Run Phase 1 successfully first."
    )

saved_model = joblib.load("best_wait_time_model.pkl")
metadata = joblib.load("project_metadata.pkl")

print("\nSaved model name:", metadata["model_name"])
print("Target column:", metadata["target_column"])
print("Number of input features:",
      len(metadata["feature_columns"]))

# 3. Verify the saved model with a test patient row
# X_test must come from the successful Phase 1 run.

if "X_test" not in globals() or "y_test" not in globals():
    raise ValueError(
        "Run the complete Phase 1 script first, "
        "then run this Phase 2 section."
    )

sample_patient = X_test.iloc[[0]]
actual_wait = float(y_test.iloc[0])

predicted_wait = float(
    saved_model.predict(sample_patient)[0]
)

print("\n========== TEST PREDICTION ==========")
print("Actual doctor waiting time:",
      round(actual_wait, 2), "minutes")
print("Predicted doctor waiting time:",
      round(predicted_wait, 2), "minutes")
print("Absolute prediction error:",
      round(abs(actual_wait - predicted_wait), 2),
      "minutes")

# 4. Confirm the prediction is valid

if np.isfinite(predicted_wait) and predicted_wait >= 0:
    print("\nSUCCESS: Saved model generated a valid prediction.")
else:
    print("\nWARNING: Check the model prediction.")

# 5. Display saved model comparison

comparison = joblib.load if False else None
results_check = __import__("pandas").read_csv(
    "model_comparison.csv"
)

print("\n========== MODEL COMPARISON ==========")
print(results_check.to_string(index=False))

print("\nPHASE 2 STEP 1 COMPLETED")


# ============================================================
# PHASE 2 - STEP 2: REUSABLE PREDICTION FUNCTION
# ============================================================

import joblib
import pandas as pd
import numpy as np

print("\n========== PHASE 2: PREDICTION FUNCTION ==========")

# 1. Load saved model and metadata

saved_model = joblib.load("best_wait_time_model.pkl")
metadata = joblib.load("project_metadata.pkl")

feature_columns = metadata["feature_columns"]
target_column = metadata["target_column"]

print("Model:", metadata["model_name"])
print("Target:", target_column)
print("Required input features:", feature_columns)


# 2. Create a reusable prediction function

def predict_wait_time(patient_data):

    # Check that input is a dictionary
    if not isinstance(patient_data, dict):
        raise TypeError(
            "Patient data must be provided as a dictionary."
        )

    # Check for missing required features
    missing_features = [
        col for col in feature_columns
        if col not in patient_data
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            + str(missing_features)
        )

    # Arrange features in the same order as model training
    input_df = pd.DataFrame(
        [patient_data],
        columns=feature_columns
    )

    # Generate prediction
    prediction = float(saved_model.predict(input_df)[0])

    # Validate the result
    if not np.isfinite(prediction) or prediction < 0:
        raise ValueError(
            "The model returned an invalid waiting-time prediction."
        )

    return round(prediction, 2)


# 3. Prepare one existing test patient as a demonstration
# This uses an existing test row, not invented patient values.

sample_patient_data = X_test.iloc[0][feature_columns].to_dict()

print("\nSample input prepared successfully.")

# 4. Call the reusable function

predicted_wait = predict_wait_time(sample_patient_data)

print("\n========== PREDICTION RESULT ==========")
print("Predicted doctor waiting time:",
      predicted_wait, "minutes")

print("\nPHASE 2 STEP 2 COMPLETED")



# ============================================================
# PHASE 2 - STEP 3: CHECK REQUIRED MODEL INPUTS
# ============================================================

import joblib

metadata = joblib.load("project_metadata.pkl")

feature_columns = metadata["feature_columns"]

print("\n========== REQUIRED PREDICTION INPUTS ==========")

for number, column in enumerate(feature_columns, start=1):
    print(f"{number}. {column}")

print("\nTotal required inputs:", len(feature_columns))
print("Prediction target:", metadata["target_column"])
print("Selected model:", metadata["model_name"])

