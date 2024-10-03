import pandas as pd
import shap
import joblib  # To load your model
from sklearn.model_selection import train_test_split

# Load your trained model
model = joblib.load('C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/EthioMart/model/my_model.joblib')  # Adjust path to your model

# Load your data
data = pd.read_csv('C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/EthioMart/data/my_data.csv')  # Adjust path to your data

# Split the data into features and target
X = data.drop(columns='target')  # Replace 'target' with your actual target column name
y = data['target']

# Optional: Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a SHAP explainer
explainer = shap.Explainer(model, X_train)

# Calculate SHAP values
shap_values = explainer(X_val)

# Plot summary of SHAP values
shap.summary_plot(shap_values, X_val)

# Optionally, plot SHAP values for a specific instance
shap.initjs()  # Initialize JavaScript in Jupyter Notebook for interactive plots
instance_index = 0  # Change to the index of the instance you want to interpret
shap.force_plot(explainer.expected_value, shap_values[instance_index], X_val.iloc[instance_index])
