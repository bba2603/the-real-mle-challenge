from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)
import pandas as pd
import numpy as np
from src.setup_logger import get_logger
from config.classifier_config import MAP_CATEGORY

class Evaluator:
    """
    A class for evaluating machine learning models.

    This class provides static methods to evaluate model performance,
    calculate feature importances, and format classification reports.

    Methods:
        evaluate(model, X_test, y_test) -> dict:
            Evaluate the model's performance on test data.
        get_feature_importances(model, training_data) -> dict:
            Get the feature importances from the model.
        get_classification_output(y_test, y_pred) -> dict:
            Get the classification report.
    """

    logger = get_logger(__name__)

    @staticmethod
    def evaluate(model, X_test, y_test) -> dict:
        """
        Evaluate the model's performance on test data.

        Args:
            model: The trained machine learning model.
            X_test: The feature matrix for testing.
            y_test: The true labels for testing.

        Returns:
            dict: A dictionary containing various evaluation metrics.
        """
        try:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
        except Exception as e:
            Evaluator.logger.error(f"Error evaluating the model: {e}")
            raise ValueError(f"Error evaluating the model: {e}")

        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': float(roc_auc_score(
                y_test, y_proba, multi_class='ovr'
            )),
            'feature_importances': Evaluator.get_feature_importances(
                model, X_test
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': Evaluator.get_classification_output(
                y_test, y_pred
            )
        }
        Evaluator.logger.info("Model evaluation completed successfully")
        Evaluator.logger.info('Accuracy: %.2f' % results['accuracy'])
        Evaluator.logger.info('ROC AUC: %.2f' % results['roc_auc'])
        Evaluator.logger.info('Feature importances: %s' % results['feature_importances'])
        Evaluator.logger.info('Confusion matrix: %s' % results['confusion_matrix'])
        Evaluator.logger.info('Classification report: %s' % results['classification_report'])
        return results

    @staticmethod
    def get_feature_importances(model, training_data) -> dict:
        """
        Calculate and sort feature importances.

        Args:
            model: The trained machine learning model.
            training_data: The feature matrix used for training.

        Returns:
            dict: A dictionary of feature names and their importance scores.
        """
        try:
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            features = training_data.columns[indices]
            importances = importances[indices].tolist()
            return dict(zip(features, importances))
        except Exception as e:
            Evaluator.logger.error(f"Error calculating feature importances: {e}")
            raise ValueError(f"Error calculating feature importances: {e}")
    
    @staticmethod
    def get_classification_output(y_test, y_pred) -> dict:
        """
        Generate a formatted classification report.

        Args:
            y_test: The true labels.
            y_pred: The predicted labels.

        Returns:
            dict: A dictionary representation of the classification report.
        """
        try:
            report = classification_report(y_test, y_pred, output_dict=True)
            df_report = pd.DataFrame.from_dict(report).T[:-3]
            df_report.index = [MAP_CATEGORY[i] for i in df_report.index]
            return df_report.to_dict()
        except Exception as e:
            Evaluator.logger.error(f"Error generating classification report: {e}")
            raise ValueError(f"Error generating classification report: {e}")
