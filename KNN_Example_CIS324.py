# Import the required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris


# Load the iris dataset that's available in scikit-learn
iris = load_iris()

# Convert to a dataframe
data = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                    columns=iris['feature_names'] + ['target'])

# Process the datasets into features and target variable, then into training and test sets
X = data.drop('target', axis = 1)
Y = data['target']

# Split into training and test sets
X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the model using K-Nearest Neighbors algorithm

# Initialize the model
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_test, Y_test)

# Making Predictions
Y_prediction = knn.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(Y_test, Y_prediction)
print(f"Accuracy ---> {accuracy:.2f}")

# Classification report
print(classification_report(Y_test, Y_prediction, target_names=iris["target_names"]))
