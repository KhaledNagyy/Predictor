# First, make sure you have installed the required package
!pip install ucimlrepo

# Import necessary libraries
import pandas as pd  # Import pandas for data manipulation
from ucimlrepo import fetch_ucirepo  # Import fetch_ucirepo function from ucimlrepo library

# Fetch dataset
heart_disease = fetch_ucirepo(id=45)  # Fetch the Heart Disease dataset from UCI ML Repository

# Get features and targets
X = heart_disease.data.features  # Extract features from the dataset
y = heart_disease.data.targets  # Extract targets from the dataset

# Convert features to a pandas DataFrame
df = pd.DataFrame(X, columns=heart_disease.feature_names)  # Create a DataFrame with features and column names

# Show column names
print("Column Names:")  # Print a message indicating column names
print(df.columns.tolist())  # Print the list of column names

# Show number of rows and columns
num_rows, num_columns = df.shape  # Get the number of rows and columns in the DataFrame
print("\nNumber of Rows:", num_rows)  # Print the number of rows
print("Number of Columns:", num_columns)  # Print the number of columns

# Display first 10 rows
print("\nFirst 10 rows:")  # Print a message indicating the display of first 10 rows
print(df.head(10))  # Display the first 10 rows of the DataFrame

# Display last 10 rows
print("\nLast 10 rows:")  # Print a message indicating the display of last 10 rows
print(df.tail(10))  # Display the last 10 rows of the DataFrame

# Display basic statistics
print("Basic Statistics:")  # Print a message indicating basic statistics
print(df.describe())  # Display basic statistics of the DataFrame using describe() method

# Check for missing values
missing_values = df.isnull().sum()  # Count the number of missing values for each column
print("\nMissing Values:")  # Print a message indicating missing values
print(missing_values)  # Print the count of missing values for each column

# Import necessary libraries
import pandas as pd  # Import pandas for data manipulation
from ucimlrepo import fetch_ucirepo  # Import fetch_ucirepo function from ucimlrepo library
from sklearn.impute import SimpleImputer  # Import SimpleImputer for handling missing values
from sklearn.compose import ColumnTransformer  # Import ColumnTransformer for preprocessing
from sklearn.preprocessing import OneHotEncoder  # Import OneHotEncoder for encoding categorical features

# Fetch dataset
heart_disease = fetch_ucirepo(id=45)  # Fetch the Heart Disease dataset from UCI ML Repository

# Get features and targets
X = heart_disease.data.features  # Extract features from the dataset
y = heart_disease.data.targets  # Extract targets from the dataset

# Convert features to a pandas DataFrame
df = pd.DataFrame(X, columns=heart_disease.feature_names)  # Create a DataFrame with features and column names

# Handle missing values
# Impute missing values in numerical features using mean
num_imputer = SimpleImputer(strategy="mean")  # Create a SimpleImputer object with mean strategy
num_features = df.select_dtypes(include=["number"]).columns  # Select numerical features
df[num_features] = num_imputer.fit_transform(df[num_features])  # Impute missing values in numerical features

# Encode categorical features (if any)
# In this dataset, there are no categorical features to encode

# Display preprocessed data
print("Preprocessed Data:")  # Print a message indicating preprocessed data
print(df.head())  # Display the first few rows of the preprocessed DataFrame

# Import necessary libraries
from sklearn.model_selection import train_test_split  # Import train_test_split function

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)  # Split the data

# Display the shapes of the training and testing sets
print("Shape of X_train:", X_train.shape)  # Print the shape of X_train (training features)
print("Shape of X_test:", X_test.shape)  # Print the shape of X_test (testing features)
print("Shape of y_train:", y_train.shape)  # Print the shape of y_train (training targets)
print("Shape of y_test:", y_test.shape)  # Print the shape of y_test (testing targets)

# Import necessary libraries
from sklearn.svm import SVC  # Import Support Vector Machine Classifier

# Initialize the Support Vector Machine Classifier
svm_classifier = SVC(random_state=42)  # Initialize SVM classifier with a random state for reproducibility

# Reshape the target variable y_train and y_test
y_train_reshaped = y_train.values.ravel()  # Reshape y_train to a 1D array
y_test_reshaped = y_test.values.ravel()  # Reshape y_test to a 1D array

# Train the model on the training data
svm_classifier.fit(X_train, y_train_reshaped)  # Fit SVM classifier to the training data

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  # Import necessary metrics

# Predict on the testing data
y_pred = svm_classifier.predict(X_test)  # Predict using the trained SVM classifier

# Calculate accuracy
accuracy = accuracy_score(y_test_reshaped, y_pred)  # Calculate accuracy score
print("Accuracy:", accuracy)  # Print the accuracy score

# Calculate precision
precision = precision_score(y_test_reshaped, y_pred, average='weighted', zero_division=1)  # Calculate precision score
print("Precision:", precision)  # Print the precision score

# Calculate recall
recall = recall_score(y_test_reshaped, y_pred, average='weighted', zero_division=1)  # Calculate recall score
print("Recall:", recall)  # Print the recall score

# Calculate F1-score
f1 = f1_score(y_test_reshaped, y_pred, average='weighted', zero_division=1)  # Calculate F1 score
print("F1 Score:", f1)  # Print the F1 score

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test_reshaped, y_pred)  # Generate confusion matrix
print("Confusion Matrix:")  # Print a message indicating confusion matrix
print(conf_matrix)  # Print the confusion matrix

# Perform necessary imports
from sklearn.model_selection import GridSearchCV  # Import GridSearchCV for hyperparameter tuning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  # Import necessary metrics
from sklearn.svm import SVC  # Import Support Vector Machine Classifier

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear'],  # Kernel type
}

# Initialize the SVM classifier
svm_classifier = SVC(random_state=42)  # Initialize SVM classifier with a random state for reproducibility

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy')  # Perform hyperparameter tuning using GridSearchCV
grid_search.fit(X_train, y_train_reshaped)  # Fit the GridSearchCV object to the training data

# Get the best hyperparameters
best_params = grid_search.best_params_  # Get the best hyperparameters from the GridSearchCV results

# Get the best model
best_model = grid_search.best_estimator_  # Get the best model from the GridSearchCV results

# Evaluate the best model
y_pred = best_model.predict(X_test)  # Predict using the best model
accuracy = accuracy_score(y_test_reshaped, y_pred)  # Calculate accuracy score
precision = precision_score(y_test_reshaped, y_pred, average='weighted', zero_division=1)  # Calculate precision score
recall = recall_score(y_test_reshaped, y_pred, average='weighted', zero_division=1)  # Calculate recall score
f1 = f1_score(y_test_reshaped, y_pred, average='weighted', zero_division=1)  # Calculate F1 score
conf_matrix = confusion_matrix(y_test_reshaped, y_pred)  # Generate confusion matrix

# Print the results
print("Best Hyperparameters:", best_params)  # Print the best hyperparameters
print("\nEvaluation Metrics of the Best Model:")  # Print a message indicating evaluation metrics
print("Accuracy:", accuracy)  # Print the accuracy score
print("Precision:", precision)  # Print the precision score
print("Recall:", recall)  # Print the recall score
print("F1 Score:", f1)  # Print the F1 score
print("Confusion Matrix:")  # Print a message indicating confusion matrix
print(conf_matrix)  # Print the confusion matrix
