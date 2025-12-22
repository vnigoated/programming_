import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. Load the dataset ---
# NOTE: The 'r' before the string handles the file path correctly on Windows
file_path = r"C:\Users\vnina\OneDrive\Desktop\ml_ese\ML_dataset\SVM_DATASET.csv"
# If you are running this with the file in the same folder, just use:
# file_path = 'SVM_DATASET.csv'

df = pd.read_csv(file_path)

# --- 2. Preprocessing ---
# Drop 'id' as it's not a feature, and 'Unnamed: 32' which is often an empty column in this dataset
if 'id' in df.columns:
    df = df.drop('id', axis=1)
if 'Unnamed: 32' in df.columns:
    df = df.drop('Unnamed: 32', axis=1)

# Encode the target 'diagnosis' (M -> 1, B -> 0)
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])
print(f"Target classes: {le.classes_}")

# Define Features (X) and Target (y)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Feature Scaling ---
# SVM is very sensitive to the scale of data, so standardization is required
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. Initialize and Train SVM ---
# kernel='linear' is often good for high-dimensional data like this, but 'rbf' is also common
model = SVC(kernel='linear', random_state=42)
model.fit(X_train_scaled, y_train)

# --- 5. Evaluate ---
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

# --- 6. Visualization ---
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('SVM Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
