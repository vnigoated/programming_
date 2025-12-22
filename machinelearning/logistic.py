import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


DATA_PATH = Path(r"C:\Users\vnina\OneDrive\Desktop\ml_ese\ML_dataset\Logistic_regression.csv")
if not DATA_PATH.exists():
	raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please check the path and try again.")

df = pd.read_csv(DATA_PATH)


X = df[['age']]
y = df['bought_insurance']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)


plt.figure(figsize=(10, 6))
plt.scatter(df['age'], df['bought_insurance'], color='blue', marker='+', label='Data Points')


X_range = np.linspace(df['age'].min() - 5, df['age'].max() + 5, 300).reshape(-1, 1)
y_prob = model.predict_proba(X_range)[:, 1] 

plt.plot(X_range, y_prob, color='red', label='Logistic Regression Curve')
plt.xlabel('Age')
plt.ylabel('Probability of Buying Insurance')
plt.title('Logistic Regression: Age vs. Buying Insurance')
plt.legend()
plt.grid(True)
plt.show()
