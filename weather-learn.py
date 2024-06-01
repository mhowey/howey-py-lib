import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
# Replace 'your_dataset.csv' with the path to your actual CSV file
data = pd.read_csv('weather_data.csv')

# Assume the last column is the target variable and the rest are features
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression()
# Train the model
# model.fit(X_train[2], y_train[2])
print(X_train)
print(y_train)

# Make predictions on the test set
# y_pred = model.predict(X_test)

# Calculate the accuracy of the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy * 100:.2f}%')