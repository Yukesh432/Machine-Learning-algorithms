from typing import Dict, Optional, Any

class ModelConfig:
    """
    Configuration schema for machine learning models.
    """
    def __init__(
            self,
            name: str,
            task_type: str,
            params: Optional[Dict[str, Any]] = None,
            random_state: Optional[int] = 42,

    ):
        self.name= name
        self.task_type = task_type
        self.params = params if params is not None else {}
        self.random_state = random_state

    def __repr__(self)-> str:
        return (
            f"ModelConfig(name={self.name}, task_type={self.task_type}, "
            f"params={self.params}, random_state={self.random_state})"
        )
    

# checking the code:
if __name__ == "__main__":
    config = ModelConfig(
        name="RandomForestClassifier",
        task_type="classification",
        params={"n_estimators": 100, "max_depth": 10},
        random_state=42
    )
    print(config)