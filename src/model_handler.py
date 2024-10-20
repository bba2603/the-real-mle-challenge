import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from config.classifier_config import (
    N_ESTIMATORS, RANDOM_STATE_CLASSIFIER, CLASS_WEIGHT, N_JOBS
)

from src.setup_logger import get_logger

class ModelHandler:
    """
    A class for handling machine learning model operations.

    This class provides methods for loading, saving, and training
    a RandomForestClassifier model.

    Attributes:
        model: The machine learning model (RandomForestClassifier).

    Methods:
        load_model(path: str) -> None:
            Load a trained model from a file.
        save_model(path: str) -> None:
            Save the current model to a file.
        train_model(X_train, y_train) -> None:
            Train a new RandomForestClassifier model with the given data.
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.model = None

    def load_model(self, path: str) -> None:
        """
        Load a trained model from a file.

        Args:
            path (str): The file path to load the model from.
        """
        # Check if the file exists
        if not os.path.exists(path):
            self.logger.error(f"The file at {path} does not exist")
            raise FileNotFoundError(f"The file at {path} does not exist")

        self.model = pickle.load(open(path, 'rb'))
        self.logger.info(f"Model loaded from {path}")

        return self.model
    
    def save_model(self, path: str) -> None:
        """
        Save the current model to a file.

        Args:
            path (str): The file path to save the model to.
        """
        pickle.dump(self.model, open(path, 'wb'))
        self.logger.info(f"Model saved to {path}")

    def train_model(self, X_train, y_train) -> None:
        """
        Train a new RandomForestClassifier model with the given data.

        Args:
            X_train: The feature matrix for training.
            y_train: The target vector for training.
        """
        self.model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE_CLASSIFIER,
            class_weight=CLASS_WEIGHT,
            n_jobs=N_JOBS
        )
        self.logger.info("Model: RandomForestClassifier, "
                         f"n_estimators: {N_ESTIMATORS}, "
                         f"random_state: {RANDOM_STATE_CLASSIFIER}, "
                         f"class_weight: {CLASS_WEIGHT}, "
                         f"n_jobs: {N_JOBS}")

        try:
            self.model.fit(X_train, y_train)
        except Exception as e:
            self.logger.error(f"Error training the model: {e}")
            raise ValueError(f"Error training the model: {e}")

        self.logger.info("Model trained successfully")
    
    def predict(self, data):
        return self.model.predict(data)
