import pytest
import pandas as pd
import numpy as np
from src.data_preparation import DataPreparation
from config.classifier_config import MAP_ROOM_TYPE, MAP_NEIGHB, FEATURE_NAMES, TARGET_COLUMN

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'neighbourhood': ['Manhattan', 'Brooklyn', 'Queens'],
        'room_type': ['Entire home/apt', 'Private room', 'Shared room'],
        'accommodates': [2, 3, 4],
        'bathrooms': [1.0, 1.5, 2.0],
        'bedrooms': [1, 2, 3],
        TARGET_COLUMN: [0, 1, 2]
    })

@pytest.fixture
def data_preparation(sample_df):
    return DataPreparation(sample_df)

def test_mapping_columns(data_preparation):
    data_preparation.mapping_columns()

    # Check if the columns are mapped correctly
    assert data_preparation.df['neighbourhood'].tolist() == [MAP_NEIGHB['Manhattan'], MAP_NEIGHB['Brooklyn'], MAP_NEIGHB['Queens']]
    assert data_preparation.df['room_type'].tolist() == [MAP_ROOM_TYPE['Entire home/apt'], MAP_ROOM_TYPE['Private room'], MAP_ROOM_TYPE['Shared room']]

def test_split_data(data_preparation):
    X_train, X_test, y_train, y_test = data_preparation.split_data()
    
    # Check if the columns are correct
    assert set(X_train.columns) == set(FEATURE_NAMES)
    assert set(X_test.columns) == set(FEATURE_NAMES)
    assert y_train.name == TARGET_COLUMN
    assert y_test.name == TARGET_COLUMN
    
    # Check if the data is split correctly
    assert len(X_train) + len(X_test) == len(data_preparation.df)
    assert len(y_train) + len(y_test) == len(data_preparation.df)