import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load your dataset
data_path = 'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/EthioMart/data/my_data.csv'  # Adjust path to your data
data = pd.read_csv(data_path)

# Display initial data information
print("Initial Data Information:")
print(data.info())

# Handle missing values (example: fill with mean for numerical columns and mode for categorical columns)
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    data[column].fillna(data[column].mean(), inplace=True)

for column in data.select_dtypes(include=['object']).columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

# Define feature columns and target column
X = data.drop(columns='target')  # Replace 'target' with your actual target column name
y = data['target']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Preprocess the data
X_preprocessed = preprocessor.fit_transform(X)

# Optionally, split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Save the preprocessed data if needed
import joblib
joblib.dump(preprocessor, 'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/EthioMart/model/preprocessor.joblib')  # Adjust path

print("Preprocessing complete. Preprocessed data and model saved.")
