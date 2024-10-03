import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def load_data(file_path):
    """
    Load CSV data from the given file path.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def save_model(model, file_path):
    """
    Save the trained model to a file.

    Args:
        model: The model to save.
        file_path (str): The file path to save the model.
    """
    import joblib
    try:
        joblib.dump(model, file_path)
        print(f"Model saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model performance using various metrics.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    print(f"Model Evaluation Metrics: {metrics}")
    return metrics

def get_feature_importances(model, feature_names):
    """
    Get feature importances from the model.

    Args:
        model: The trained model.
        feature_names (list): List of feature names.

    Returns:
        pd.DataFrame: DataFrame containing feature importances.
    """
    importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
    if importances is not None:
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        return feature_importance_df
    else:
        print("The model does not have feature importances.")
        return None

def set_random_seed(seed=42):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    import random
    import os
    import tensorflow as tf
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    print(f"Random seed set to {seed}")

