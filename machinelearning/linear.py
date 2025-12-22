import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
# Use a raw string for the Windows path so backslashes aren't interpreted as escape sequences.
df = pd.read_csv(r"C:\Users\vnina\OneDrive\Desktop\ml_ese\ML_dataset\linear_regression.csv")

# 2. Prepare the data
# 'year' is our independent variable (X)
# 'per capita income (US$)' is our dependent variable (y)
X = df[['year']]
y = df['per capita income (US$)']

# 3. Split the data into training and testing sets
# We use 20% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = model.predict(X_test)

# 6. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Coefficients: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# 7. Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.xlabel('Year')
plt.ylabel('Per Capita Income (US$)')
plt.title('Year vs Per Capita Income Linear Regression')
plt.legend()
plt.show()
