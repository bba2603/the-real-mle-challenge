import pytest
import numpy as np
import pandas as pd
from src.model_evaluator import Evaluator

@pytest.fixture
def mock_model():
    class MockModel:
        def predict(self, X):
            return np.array([1, 2, 3, 0])
        
        def predict_proba(self, X):
            return np.array([[0.64948664, 0.25335874, 0.04964917, 0.04750545],
                             [0.03626272, 0.53504922, 0.42868805, 0.        ],
                             [0.14805084, 0.54754792, 0.26353927, 0.04086197],
                             [0.1662008 , 0.66404412, 0.16974908, 0.0        ]])
        
        feature_importances_ = np.array([0.6, 0.4])
    
    return MockModel()

def test_evaluate(mock_model):
    X_test = pd.DataFrame({'feature1': [1, 2, 3, 4], 'feature2': [5, 6, 7, 8]})
    y_test = pd.Series([1, 2, 3, 0], name='category')
    
    results = Evaluator.evaluate(mock_model, X_test, y_test)
    
    # Check if the metrics of results are calculated without errors
    assert 'accuracy' in results
    assert 'roc_auc' in results
    assert 'feature_importances' in results
    assert 'confusion_matrix' in results
    assert 'classification_report' in results