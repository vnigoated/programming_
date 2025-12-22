import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# --- 1. Load the dataset ---
# NOTE: Use 'r' before the path string if you are on Windows
file_path = r"C:\Users\vnina\OneDrive\Desktop\ml_ese\ML_dataset\k-means-clustring.csv"
# file_path = 'k-means-clustring.csv' # Use this if file is in same folder
df = pd.read_csv(file_path)

# --- 2. Preprocessing (Scaling) ---
# K-Means is sensitive to scale. Income (e.g., 150,000) is much larger than Age (e.g., 40).
# If we don't scale, Income will dominate the distance calculation.
scaler = MinMaxScaler()

df['Age_scaled'] = scaler.fit_transform(df[['Age']])
df['Income_scaled'] = scaler.fit_transform(df[['Income($)']])

# --- 3. Find Optimal K (Elbow Method) ---
sse = [] # Sum of Squared Errors
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(df[['Age_scaled', 'Income_scaled']])
    sse.append(km.inertia_)

# Plot Elbow (Optional, to verify k)
plt.figure(figsize=(6, 4))
plt.plot(k_rng, sse)
plt.xlabel('K')
plt.ylabel('Sum of Squared Error')
plt.title('Elbow Plot')
plt.show()

# --- 4. Train Model with Optimal K ---
# Based on the elbow plot, k=3 is usually best for this dataset
km = KMeans(n_clusters=3, n_init=10, random_state=42)
y_predicted = km.fit_predict(df[['Age_scaled', 'Income_scaled']])

# Add cluster result to dataframe
df['cluster'] = y_predicted
print(df.head())

# --- 5. Visualization ---
# Separate dataframes for each cluster for easy plotting
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.figure(figsize=(8, 6))

# Plot data points
plt.scatter(df1.Age_scaled, df1.Income_scaled, color='green', label='Cluster 1')
plt.scatter(df2.Age_scaled, df2.Income_scaled, color='red', label='Cluster 2')
plt.scatter(df3.Age_scaled, df3.Income_scaled, color='black', label='Cluster 3')

# Plot centroids
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], 
            color='purple', marker='*', label='Centroid', s=150)

plt.xlabel('Age (Scaled)')
plt.ylabel('Income (Scaled)')
plt.title('K-Means Clustering (k=3)')
plt.legend()
plt.show()
