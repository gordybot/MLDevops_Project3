from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : 
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def performance_on_slice(
        test, feature, model, data_processor):
    """
    Calculate and write to file 
    the performance on a 'slice' of the data - 
    for only a particular categorical feature.

    Inputs
    ------
    test: pd.DataFrame
          X has features and y has the output.
    feature: str
        feature to slice
    model: 
        Trained machine learning model.
    data_preprocessor:  func
        function to preprocess the data.
    Returns
    -------
    None
    """
    with open(f"model/{feature}_slice_performance.txt", "w") as f:
        f.write(f"Performance for {feature} category.")

        feature_set = set(test[feature])
        for feature_slice in feature_set:
            df = test[ test[feature]==feature_slice ]
            X_test, y_test, _, _ = process_data_Func(df=df)
            predictions = inference(model, X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, predictions)

            f.write("\n")
            f.write(f"{feature_slice}:\n")
            f.write(f"fbeta:     {fbeta}\n")
            f.write(f"precision:      {precision}\n")
            f.write(f"recall:        {recall}\n")
