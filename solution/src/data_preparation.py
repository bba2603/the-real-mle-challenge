import pandas as pd
from sklearn.model_selection import train_test_split
from config.classifier_config import (
    MAP_ROOM_TYPE, MAP_NEIGHB, FEATURE_NAMES, SPLIT_RATIO, RANDOM_STATE_SPLIT
)

from src.setup_logger import get_logger

class DataPreparation:
    """
    A class for preparing and processing data for the Airbnb price category classifier.

    This class handles data mapping and splitting operations on the input DataFrame.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing Airbnb listing data.

    Methods:
        mapping_columns(self) -> None:
            Map categorical columns to numerical values.
        split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            Split the data into training and testing sets.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.logger = get_logger(__name__)
        self.df = df

    def mapping_columns(self) -> None:
        """
        Map categorical columns to numerical values.

        This method applies predefined mappings to the 'neighbourhood' and 'room_type' columns,
        converting categorical values to numerical representations.
        """
        try:
            self.df["neighbourhood"] = self.df["neighbourhood"].map(MAP_NEIGHB)
            self.df["room_type"] = self.df["room_type"].map(MAP_ROOM_TYPE)
        except Exception as e:
            self.logger.error(f"Error mapping columns: {e}")
            raise ValueError(f"Error mapping columns: {e}")

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the data into training and testing sets.

        This method separates the feature columns (X) from the target column (y),
        and then splits the data into training and testing sets using predefined parameters.

        Returns:
            tuple: A tuple containing X_train, X_test, y_train, y_test
        """
        X = self.df[FEATURE_NAMES]
        y = self.df['category']
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=SPLIT_RATIO, random_state=RANDOM_STATE_SPLIT
            )
        except Exception as e:
            self.logger.error(f"Error splitting data: {e}")
            raise ValueError(f"Error splitting data: {e}")

        self.logger.info(f"Data split. (X_train: {X_train.shape}, "
                         f"X_test: {X_test.shape}, y_train: {y_train.shape}, "
                         f"y_test: {y_test.shape})")
        return X_train, X_test, y_train, y_test