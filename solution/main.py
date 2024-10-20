from config.preprocessing_config import PROCESSED_FOLDER
from config.classifier_config import MODEL_FOLDER, RESULTS_FOLDER

from src.preprocessor import DataProcessor
from src.data_preparation import DataPreparation
from src.model_handler import ModelHandler
from src.model_evaluator import Evaluator
from src.setup_logger import setup_logger, get_logger

import os
import json
from datetime import datetime
from pathlib import Path

def main():
    # Get the current time for unique file naming
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup logger
    setup_logger(current_time)
    logger = get_logger(__name__)

    # Get SRC_PATH from environment variable
    SRC_PATH = os.environ.get('SRC_PATH')
    if SRC_PATH:
        logger.info(f"SRC_PATH is set to: {SRC_PATH}")
    else:
        SRC_PATH = "data/raw/listings.csv"
        logger.info(f"SRC_PATH is not set. Using default path: {SRC_PATH}")

    # Initialize the data processor and process the raw data
    data_processor = DataProcessor()
    data_processor.load_data(SRC_PATH)
    processed_path = Path(PROCESSED_FOLDER) / \
        f'processed_listings_{current_time}.csv'
    data_processor.process_data(processed_path)

    # Prepare the processed data for model training
    data_prep = DataPreparation(data_processor.df)
    data_prep.mapping_columns()  # Map categorical columns to numerical values
    X_train, X_test, y_train, y_test = data_prep.split_data()

    # Initialize the model handler and train the model
    model_handler = ModelHandler()
    model_handler.train_model(X_train, y_train)

    # Evaluate the trained model
    results = Evaluator.evaluate(model_handler.model, X_test, y_test)

    # Save the trained model
    model_path = Path(MODEL_FOLDER) / f'model_{current_time}.pkl'
    model_handler.save_model(model_path)

    # Save the evaluation results
    results_path = Path(RESULTS_FOLDER) / f'results_{current_time}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
