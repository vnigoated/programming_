import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


USING_XGBOOST = False
try:
    import xgboost as xgb
    USING_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    print("XGBoost not found. Using sklearn GradientBoostingClassifier. Install with: pip install xgboost")

# Load dataset
file_path = r"C:\Users\vnina\OneDrive\Desktop\ml_ese\ML_dataset\XG-boost.csv"
df = pd.read_csv(file_path)

# Safely drop identifier columns if they exist
drop_cols = [c for c in ("Transaction_ID", "User_ID") if c in df.columns]
if drop_cols:
    df.drop(columns=drop_cols, inplace=True)

# Handle timestamp if present
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df.drop(columns=['Timestamp'], inplace=True)

# One-hot encode existing categorical columns
categorical_cols = ['Transaction_Type', 'Device_Type', 'Location', 'Merchant_Category', 'Card_Type', 'Authentication_Method']
existing_cat_cols = [col for col in categorical_cols if col in df.columns]
if existing_cat_cols:
    df = pd.get_dummies(df, columns=existing_cat_cols, drop_first=True)

# Ensure target column exists
target_col = 'Fraud_Label'
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found in dataset.")

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
if USING_XGBOOST:
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
else:
    model = GradientBoostingClassifier(random_state=42)

print("Training model...")
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

# Plot feature importance (top 15)
plt.figure(figsize=(12, 8))
top_n = 15
if USING_XGBOOST:
    xgb.plot_importance(model, max_num_features=top_n)
    plt.title('XGBoost Feature Importance')
else:
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    sorted_idx_top = sorted_idx[-top_n:]
    plt.barh(range(len(sorted_idx_top)), feature_importance[sorted_idx_top], align='center')
    plt.yticks(range(len(sorted_idx_top)), np.array(X.columns)[sorted_idx_top])
    plt.xlabel('Feature Importance')
    plt.title('Gradient Boosting Feature Importance (Top 15)')

plt.tight_layout()
plt.show()

