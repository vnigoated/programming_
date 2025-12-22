import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. Load the dataset ---
# NOTE: Use 'r' before the path string if you are on Windows to avoid errors
file_path = r"C:\Users\vnina\OneDrive\Desktop\ml_ese\ML_dataset\bagging.csv"
# file_path = 'bagging.csv' # Use this if the file is in the same folder
df = pd.read_csv(file_path)

# --- 2. Preprocessing ---
# In this dataset, 0 in these columns indicates missing data, not a real value.
# We replace 0 with NaN and then fill with the mean.
cols_with_missing = ['Glu   cose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_missing] = df[cols_with_missing].replace(0, np.nan)
df.fillna(df.mean(), inplace=True)

# Define Features (X) and Target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (helps with convergence and performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Initialize and Train Bagging Model ---
# We use a Decision Tree as the base estimator
base_cls = DecisionTreeClassifier(random_state=42)

# n_estimators=100 creates an ensemble of 100 trees
model = BaggingClassifier(estimator=base_cls, n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# --- 4. Evaluate ---
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

# --- 5. Visualization ---
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Bagging Classifier Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

