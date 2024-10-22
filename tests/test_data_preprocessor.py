import pytest
import pandas as pd
import numpy as np
from src.data_preprocessor import DataProcessor
from config.preprocessing_config import COLUMNS, RENAMED_COLUMNS, FEATURE_AMENITIES, TARGET_COLUMN

@pytest.fixture
def data_processor():
    return DataProcessor()

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id': [1, 2, 3],
        'neighbourhood_group_cleansed': ['Manhattan', 'Brooklyn', 'Queens'],
        'property_type': ['Entire rental unit', 'Private room in rental unit', 'Private room in residential home'],
        'room_type': ['Entire home/apt', 'Private room', 'Shared room'],
        'latitude': [40.7, 40.8, 40.9],
        'longitude': [-73.9, -74.0, -74.1],
        'accommodates': [2, 3, 4],
        'bathrooms': ['NaN', 'NaN', 'NaN'],
        'bathrooms_text': ['1 bath', '1.5 baths', '2 shared bath'],
        'bedrooms': [1.0, 2.0, 3.0],
        'beds': [1.0, 2.0, 3.0],
        'amenities': ['["Long term stays allowed", "Hot water", "Heating", "Wifi", "Kitchen", "Microwave"]', '["Hangers", "Kitchen", "Long term stays allowed", "Carbon monoxide alarm", "Lock on bedroom door", "Dedicated workspace", "Wifi", "Heating", "Smoke alarm"]', '["TV", "Indoor fireplace", "First aid kit", "Hangers", "Long term stays allowed", "Carbon monoxide alarm", "Wifi", "Heating", "Dishes and silverware", "Shampoo", "Air conditioning", "Essentials", "Hot water", "Kitchen", "Cooking basics", "Dedicated workspace", "Stove", "Smoke alarm", "Oven", "Refrigerator", "Smart lock", "Coffee maker"]'],
        'price': ['$100.00', '$200.00', '$300.00']
    })

def test_load_data(data_processor, sample_df, tmp_path):
    csv_path = tmp_path / "test_data.csv"
    sample_df.to_csv(csv_path, index=False)
    
    data_processor.load_data(str(csv_path))

    # Check if the dataframe is loaded
    assert data_processor.df is not None
    # Check the number of rows
    assert len(data_processor.df) == len(sample_df)

def test_clean_bathrooms_column(data_processor, sample_df):
    data_processor.df = sample_df
    data_processor.clean_bathrooms_column()

    # Check if the bathrooms column is re-created after being dropped
    assert 'bathrooms' in data_processor.df.columns
    # Check the values of the bathrooms column
    assert data_processor.df['bathrooms'].tolist() == [1.0, 1.5, 2.0]

def test_prepare_price_column(data_processor, sample_df):
    data_processor.df = sample_df
    data_processor.prepare_price_column()

    # Check if the target column is created
    assert TARGET_COLUMN in data_processor.df.columns
    # Check if the price column is converted to int
    assert data_processor.df['price'].dtype == int
    # Check if the price values are greater than or equal to 10
    assert all(data_processor.df['price'] >= 10)
    # Check if the target column values are within the expected range
    assert set(data_processor.df[TARGET_COLUMN].unique()) <= {0, 1, 2, 3}

def test_preprocess_amenities_column(data_processor, sample_df):
    data_processor.df = sample_df
    data_processor.preprocess_amenities_column()

    # Check if the amenities columns are created
    for amenity in FEATURE_AMENITIES:
        assert amenity in data_processor.df.columns

    # Check if the amenities column is dropped
    assert 'amenities' not in data_processor.df.columns

def test_process_data(data_processor, sample_df, tmp_path):
    input_path = tmp_path / "input_data.csv"
    output_path = tmp_path / "output_data.csv"
    sample_df.to_csv(input_path, index=False)
    
    data_processor.process_data(str(input_path), str(output_path))

    # Check if the output file exists
    assert output_path.exists()
    # Check if the processed dataframe has the expected columns
    processed_df = pd.read_csv(output_path)
    # Check if the amenities column is dropped. Taking in count renaming from neighbourhood_group_cleansed to neighbourhood
    columns_to_check = set(COLUMNS) - {'amenities'} - {'bathrooms_text'} - {'neighbourhood_group_cleansed'} | set(FEATURE_AMENITIES) | {TARGET_COLUMN} | {'neighbourhood'}
    assert set(processed_df.columns) == columns_to_check