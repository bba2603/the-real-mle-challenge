import pytest
import numpy as np
from sklearn.dummy import DummyClassifier
from src.model_handler import ModelHandler

@pytest.fixture
def model_handler():
    return ModelHandler()

def test_load_model(model_handler, tmp_path):
    dummy_model = DummyClassifier()
    model_path = tmp_path / "test_model.pkl"
    model_handler.model = dummy_model
    model_handler.save_model(str(model_path))
    
    loaded_model = model_handler.load_model(str(model_path))
    
    # Check if the model is loaded correctly
    assert loaded_model is not None
    assert model_handler.model is not None

def test_save_model(model_handler, tmp_path):
    model_handler.model = DummyClassifier()
    save_path = tmp_path / "saved_model.pkl"
    
    model_handler.save_model(str(save_path))
    
    # Check if the file exists
    assert save_path.exists()

def test_train_model(model_handler):
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])
    
    model_handler.train_model(X_train, y_train)
    
    # Check if the model is trained correctly (not None and has predict method)
    assert model_handler.model is not None
    assert hasattr(model_handler.model, 'predict')

def test_predict(model_handler):
    X_dummy = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_dummy = np.array([0, 1, 0, 1])
    dummy_classifier = DummyClassifier(strategy="stratified", random_state=42)
    dummy_classifier.fit(X_dummy, y_dummy)
    model_handler.model = dummy_classifier
    
    X_test = np.array([[1, 2], [3, 4]])
    predictions = model_handler.predict(X_test)
    
    # Check if the predictions are correct
    assert predictions is not None
    assert len(predictions) == len(X_test)
    
    # Check if the predictions are either 0 or 1
    assert set(predictions).issubset({0, 1})