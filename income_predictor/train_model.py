# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from income_predictor.ml.data import process_data
from income_predictor.ml.model import (train_model, 
        compute_model_metrics,
        inference)

# Add code to load in the data.
data_file_path = '../data/census_no_spaces.csv'
data = pd.read_csv(data_file_path)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False)

# Train and save a model.
model = train_model( X_train, y_train )

#  
