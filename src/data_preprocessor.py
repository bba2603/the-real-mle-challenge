import os
import pandas as pd
import numpy as np

from config.preprocessing_config import (
    COLUMNS, RENAMED_COLUMNS, FEATURE_AMENITIES, TARGET_COLUMN
)

from src.setup_logger import get_logger

class DataProcessor:
    """
    DataProcessor class to clean and preprocess the data.

    This class provides methods to load, clean, preprocess, and save data from a CSV file.
    It handles various data cleaning tasks such as fixing the bathrooms column,
    preparing the price column, and preprocessing the amenities column.

    Attributes:
        df (pd.DataFrame): The DataFrame to be processed.

    Methods:
        load_data(path: str) -> None:
            Load data from a CSV file into the DataFrame.
        
        clean_bathrooms_column() -> None:
            Clean the bathrooms column by extracting the number of bathrooms from the text.
        
        prepare_price_column() -> None:
            Prepare the price column by converting it to int, removing outliers, and creating a categorical column.
        
        preprocess_amenities_column() -> None:
            Extract categorical columns from the amenities column and create binary features.
        
        save_data(path: str) -> None:
            Save the processed DataFrame to a CSV file.
        
        process_data(input_path: str, output_path: str) -> None:
            Process the data from input to output, applying all preprocessing steps.
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.df = None

    def load_data(self, path: str) -> None:
        """
        Load data from a CSV file into the DataFrame.

        Args:
            path (str): The file path of the CSV to be loaded.
        """
        # Check if the file exists
        if not os.path.exists(path):
            self.logger.error(f"The file at {path} does not exist")
            raise FileNotFoundError(f"The file at {path} does not exist")

        self.df = pd.read_csv(path)

        # Check if the file is empty
        if self.df.empty:
            self.logger.error("The file is empty")
            raise ValueError("The file is empty")
        else:
            self.logger.info(f"File loaded. Shape of df: {self.df.shape}")

    def clean_bathrooms_column(self) -> None:
        """
        Clean the bathrooms column by extracting the number of bathrooms from the text.

        This method drops the original 'bathrooms' column and creates a new 'bathrooms' column
        with numeric values extracted from the 'bathrooms_text' column.
        """
        self.df.drop(columns=['bathrooms'], inplace=True)

        def num_bathroom_from_text(text):
            try:
                if isinstance(text, str):
                    bath_num = text.split(" ")[0]
                    return float(bath_num)
                else:
                    return np.nan
            except ValueError:
                return np.nan

        self.df['bathrooms'] = self.df['bathrooms_text'].apply(num_bathroom_from_text)
    
    def prepare_price_column(self) -> None:
        """
        Prepare the price column by converting it to int, removing outliers, and creating a categorical column.

        This method performs the following steps:
        1. Converts the 'price' column to integer values.
        2. Removes outliers by filtering out prices less than 10.
        3. Creates a new 'category' column based on price ranges.
        """
        # Convert price to int
        try:
            self.df['price'] = self.df['price'].str.extract(r"(\d+).").astype(int)
        except Exception as e:
            self.logger.error(f"Error converting price to int: {e}")
            raise ValueError(f"Error converting price to int: {e}")

        # Remove outliers
        self.df = self.df[self.df['price'] >= 10]        

        # Create categorical column
        self.df[TARGET_COLUMN] = pd.cut(
            self.df['price'],
            bins=[10, 90, 180, 400, np.inf],
            labels=[0, 1, 2, 3]
        )

    def preprocess_amenities_column(self) -> None:
        """
        Extract categorical columns from the amenities column and create binary features.

        This method creates new binary columns for each amenity in the predefined list,
        indicating the presence (1) or absence (0) of the amenity. The original 'amenities'
        column is then dropped.
        """
        for amenity in FEATURE_AMENITIES:
            self.df[amenity.replace(' ', '_')] = self.df['amenities'].str.contains(amenity).astype(int)

        self.df.drop('amenities', axis=1, inplace=True)
    
    def save_data(self, path: str) -> None:
        """
        Save the processed DataFrame to a CSV file.

        Args:
            path (str): The file path where the processed data will be saved.
        """
        self.df.to_csv(path, index=False)
        self.logger.info(f"Data saved to {path}")

    def process_data(self, input_path: str, output_path: str) -> None:
        """
        Process the data from input to output, applying all preprocessing steps.

        This method orchestrates the entire data processing pipeline by calling
        other methods in the appropriate order.

        Args:
            input_path (str): The file path of the input CSV.
            output_path (str): The file path where the processed data will be saved.
        """
        self.load_data(input_path)

        # Create bathrooms column based on bathrooms_text
        self.clean_bathrooms_column()

        # Select and rename columns
        self.df = self.df[COLUMNS]
        self.df.rename(columns=RENAMED_COLUMNS, inplace=True)

        # Prepare price: Convert it to int, remove outliers and as categorical
        self.prepare_price_column()

        # Extract categorical columns from the amenities column
        self.preprocess_amenities_column()

        # Remove NaN values
        self.df.dropna(axis=0, inplace=True)

        self.save_data(output_path)