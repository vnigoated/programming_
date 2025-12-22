import bentoml
from sklearn import svm
from sklearn import datasets

# Load dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train SVM model
model = svm.SVC(gamma='scale')          
model.fit(X, y)
print("Model trained successfully.")

# Save the trained model with BentoML
saved_model = bentoml.sklearn.save_model("svm_iris_model", model)
print("Model saved successfully with BentoML.")
print(f"Model tag: {saved_model.tag}")
print(f"Model path: {saved_model.path}")