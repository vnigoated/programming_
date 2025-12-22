import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

iris_clf_runner = bentoml.sklearn.get("svm_iris_model:latest").to_runner()
svc = bentoml.Service("iris_clf_service", runners=[iris_clf_runner])
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify_iris(input_data: np.ndarray) -> np.ndarray:
    result = iris_clf_runner.predict.run(input_data)
    return result
