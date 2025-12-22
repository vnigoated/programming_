import bentoml

iris_clf_runner = bentoml.sklearn.get("svm_iris_model:latest").to_runner()
iris_clf_runner.init_local()
print(iris_clf_runner.predict.run([[5.1, 3.5, 1.4, 0.2]]))  # Example input for prediction