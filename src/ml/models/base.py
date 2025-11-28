"""
Base class for ML models.
"""

import os

import joblib

from core.interfaces import MLPredictor


class BaseMLPredictor(MLPredictor):
    """
    Base implementation of MLPredictor.
    """

    def __init__(self, name: str):
        self._name = name
        self._model = None
        self._is_trained = False
        self._feature_names = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def save(self, path: str) -> None:
        """Save model using joblib."""
        if not self._is_trained:
            raise ValueError("Cannot save untrained model")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "feature_names": self._feature_names,
                "name": self._name,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model using joblib."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)
        self._model = data["model"]
        self._feature_names = data["feature_names"]
        self._name = data.get("name", self._name)
        self._is_trained = True
