import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Load the dataset ---
file_path = r"C:\Users\vnina\OneDrive\Desktop\ml_ese\ML_dataset\k-nearest-neighbor.csv"
# file_path = 'k-nearest-neighbor.csv' # Use this if file is in the same folder
df = pd.read_csv(file_path)

# --- 2. Preprocessing ---
# Replace '?' with NaN (standard missing value marker in Python)
df.replace('?', np.nan, inplace=True)

# Drop rows where the target 'price' is missing (we can't train on these)
df.dropna(subset=['price'], inplace=True)

# Convert numeric columns that were loaded as text/objects
numeric_cols = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm', 'price']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Impute missing values
# Fill numeric columns with the mean
cols_to_impute = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm']
for col in cols_to_impute:
    df[col].fillna(df[col].mean(), inplace=True)

# Fill 'num-of-doors' with the most common value (mode)
df['num-of-doors'].fillna(df['num-of-doors'].mode()[0], inplace=True)

# Define features and target
X = df.drop('price', axis=1)
y = df['price']

# Convert categorical variables (strings) into numbers (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# --- 3. Split and Scale ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling is CRITICAL for KNN because it calculates distances
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. Train Model ---
# We found k=1 to be the best for this specific split, but you can change n_neighbors
k = 1
model = KNeighborsRegressor(n_neighbors=k)
model.fit(X_train_scaled, y_train)

# --- 5. Evaluate ---
y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Performance (k={k}):")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# --- 6. Visualize ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='green', alpha=0.7, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'KNN Regression (k={k}): Actual vs Predicted Prices')
plt.legend()
plt.show()