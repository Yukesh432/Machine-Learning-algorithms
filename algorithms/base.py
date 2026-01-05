from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

from sklearn.base import BaseEstimator
from algorithms.schema import ModelConfig

class BaseMLModel(ABC):
    """
    Abstract base class for machine learning models.
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[BaseEstimator]= None

        self.is_built= False
        self.is_fitted= False

        # storing the evaluation results for later comparison
        self.history: List[Dict[str, Any]]= []

    @abstractmethod
    def build(self)->BaseEstimator:
        """
        Create sklearn model estimator and assign to self.model
        """
        raise NotImplementedError
    
    def fit(self, X, y=None):
        if not self.is_built or self.model is None:
            self.is_built= True
            self.build()

        if self.config.task_type!="clustering" and y is None:
            raise ValueError("y cannot be None for supervised tasks")
        
        if self.config.task_type=="clustering":
            self.model.fit(X)
        else:
            self.model.fit(X, y)

        self.is_fitted= True
        return self
    
    def predict(self, X):
        self._check_ready()
        return self.model.predict(X)
    
    def predict_proba(self, X):
        self._check_ready()
        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError(
                f"{self.config.name}: predict_proba not implemented for this model"
            )
        return self.model.predict_proba(X)


    def _check_ready(self):
        if self.model is None or not self.is_fitted:
            raise RuntimeError(
                f"{self.config.name}: model not ready (build + fit first)"
            )
        
    def record_result(self, result: Dict[str, Any]):
        """Store evaluation results (used for ranking later)."""
        self.history.append(result)
