# Our best model is Pipeline C:
# Random Forest Classifier + Random Forest Regressor

################################################
#                    Import                    #
################################################

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OrdinalEncoder,
)

################################################
#               Prepare data                   #
################################################

# Data folder
data = Path(__file__).parent / "data"

# Read data
df_train = pd.read_csv(data / "train.csv", index_col=0)
df_test = pd.read_csv(data / "test.csv", index_col=0)

# Drop duplicates
df_train = df_train.drop_duplicates()

# Valid feature ranges
valid_ranges = {
    "age": (18, 90),
    "blood_pressure_sys": (90, 182),
    "sleep_duration_hrs": (3, 12),
    "sleep_quality_score": (1, 10),
    "cardio_mins_wk": (0, 500),
    "water_intake_liters": (0.5, 5),
    "screen_time_recreational_hrs": (0, 16),
    "social_media_hrs": (0, 14),
    "diet_ultra_processed_pct": (0, 100),
    "diet_omega3_freq": (0, 7),
    "alcohol_drinks_wk": (0, 40),
    "novel_learning_hrs_wk": (0, 20),
    "social_connection_score": (1, 10),
    "chronic_stress_score": (1, 10)
}

# Features with one decimal place
one_decimal = ["social_media_hrs" , "novel_learning_hrs_wk"]


# Filtering function
def invalid_to_nan(X, ranges, one_decimal):
    X = X.copy()

    # range check
    for col, (low, high) in ranges.items():
        X.loc[~X[col].between(low, high), col] = np.nan

    # decimal places check
    for col in one_decimal:
        s = X[col]
        X.loc[~np.isclose(s * 10, np.round(s * 10)), col] = np.nan

    return X

cleaner = FunctionTransformer(
    invalid_to_nan,
    kw_args={"ranges": valid_ranges, "one_decimal": one_decimal},
    feature_names_out="one-to-one", # we do not add new columns
)

# Feature types
num_features = list(df_train.columns.drop(["smoking_status", "target"]))
cat_features = ["smoking_status"]


# Training data
X_train = df_train.drop(columns=["target"])
y_train = df_train["target"]

# Test data
X_test = df_test

################################################
#               Prepare pipeline               #
################################################

# Preprocessing of numerical features
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

# Preprocessing of categorical features
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[["Never", "Former", "Current"]])),
])

# Combined preprocessing
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# Pipeline for classification
cls_model = Pipeline([
    ("cleaner", cleaner),
    ("preprocessor", preprocessor),
    ("rfc", RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=20,
        max_samples=0.6,
        max_features=0.5,
        max_depth=None,
        class_weight=None,
        random_state=42
        )
    )
])

# Pipeline for regression
reg_model = Pipeline([
    ("cleaner", cleaner),
    ("preprocessor", preprocessor),
    ("rfr", RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=10,
        max_samples=0.9,
        max_features=0.7,
        max_depth=20,
        random_state=42
        )
    )
])


# Combined prediction
def sequential_predict(cls_model, reg_model, X):
    below_30 = cls_model.predict(X) == 0

    y = np.full(len(X), 30.0)
    if below_30.any():
        y[below_30] = reg_model.predict(X[below_30])

    return y

# Cross-validation for out-of-fold predictions
cls_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

################################################
#                  Training                    #
################################################
y_30_train = (y_train == 30).astype(int)

# Train classifier
cls_model.fit(X_train, y_30_train)

# Out-of-fold predictions from the classifier
y_30_oof = cross_val_predict(cls_model, X_train, y_30_train, cv=cls_cv, n_jobs=-1)

X_lt30_train = X_train[y_30_oof == 0]
y_lt30_train = y_train[y_30_oof == 0]

# Train regressor
reg_model.fit(X_lt30_train, y_lt30_train)

################################################
#                  Prediction                  #
################################################

y_predict = sequential_predict(cls_model, reg_model, X_test)
y_predict = np.round(y_predict)

################################################
#                    Kaggle                    #
################################################

df_sub = pd.DataFrame({"target": y_predict}, index=X_test.index)
df_sub.to_csv(data / "cls_reg_sub.csv", index=True)