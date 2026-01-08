from algorithms.base import BaseMLModel
from algorithms.schema import ModelConfig
import os

import joblib
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing, load_diabetes

from typing import Optional, Dict, Any

class LinearRegressionModel(BaseMLModel):

    def __init__(self, config: Optional[ModelConfig] = None):
        if config is None:
            config = ModelConfig(
                name="LinearRegression",
                task_type="regression",
                params={},
                random_state=42
            )
        super().__init__(config)

    def build(self) -> BaseEstimator:
        lr_params: Dict[str, Any] = self.config.params.copy()
        estimator= LinearRegression(**lr_params)

        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("estimator", estimator)
            ]
        )
        self.is_built = True
        return self.model
    

def load_regression_dataset()-> Tuple[np.ndarray, np.ndarray, str]:
    ds= load_diabetes()
    # ds= fetch_california_housing()

    X= ds.data
    y= ds.target
    # target_name= ds.target_names[0]
    dataset_name= "CaliforniaHousing"
    return X, y, dataset_name


def train_evaluate_save(
        model:LinearRegressionModel,
        models_dirs: str= "models",
        test_size: float= 0.2,
        random_state: int= 42,
)-> Dict[str, Any]:
    X, y, dataset_name= load_regression_dataset()

    X_train, X_test, y_train, y_test= train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model.fit(X_train, y_train)

    y_pred= model.predict(X_test)

    # evaluate

    mae= np.mean(np.abs(y_test - y_pred))
    mse= np.mean((y_test - y_pred)**2)
    rmse= np.sqrt(mse)
    r2= 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
    
    results = {
        "model": model.config.name,
        "task_type": model.config.task_type,
        "dataset": dataset_name,
        "test_size": test_size,
        "metrics": {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
        },
        # "params": model.get_params(),
    }

    # save
    os.makedirs(models_dirs, exist_ok=True)
    save_path= os.path.join(models_dirs, f"{model.config.name}_{dataset_name}.joblib")
    joblib.dump(model, save_path)

    # Record history inside model (useful later for ranking tables)
    model.record_result({**results, "saved_path": save_path})

    return {**results, "saved_path": save_path}


if __name__ == "__main__":
    lr_model= LinearRegressionModel()
    out= train_evaluate_save(lr_model)
    print("Training and evaluation results:")
    print("------------------------------")
    print(out)

