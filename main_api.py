from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from pathlib import Path
import pandas as pd
from datetime import datetime
import traceback

from config.classifier_config import MODEL_FOLDER, FEATURE_NAMES, MAP_CATEGORY
from src.model_handler import ModelHandler
from src.data_preparation import DataPreparation
from src.setup_logger import setup_logger, get_logger

app = FastAPI()

class ListingInput(BaseModel):
    id: int
    accommodates: int
    room_type: str
    beds: int
    bedrooms: int
    bathrooms: int
    neighbourhood: str
    tv: int
    elevator: int
    internet: int
    latitude: float
    longitude: float

class ModelToLoad(BaseModel):
    model_path: str

@app.post("/predict")
def predict_price_category(input_data: ListingInput, model_file: ModelToLoad):
    # Get the current time for unique file naming
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup logger
    setup_logger(current_time)
    logger = get_logger(__name__)

    try:
        model_path = Path(MODEL_FOLDER) / model_file.model_path
        # Check if the model file exists
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Model file not found: {model_path}"
            )

        logger.info(f"Loading model from {model_path}")

        model_handler = ModelHandler()
        model = model_handler.load_model(model_path)

        # Convert input data to a pandas DataFrame
        data = pd.DataFrame([input_data.model_dump()])
        logger.info(f"Data: \n{data}")

        # Check if all required columns are present
        missing_columns = set(FEATURE_NAMES) - set(data.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_columns}")

        # Select only the required columns
        data = data[FEATURE_NAMES]

        # Map the columns to the correct values
        data_prep = DataPreparation(data)
        data_prep.mapping_columns()
        logger.info(f"Data: \n{data_prep.df}")

        # Make prediction and map to category
        prediction = model.predict(data_prep.df)
        predicted_category = MAP_CATEGORY[str(prediction[0])].capitalize()
        logger.info(f"Prediction: {prediction[0]}")
        logger.info(f"Predicted category: {predicted_category}")

        return {"id": input_data.id, "price_category": predicted_category}

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
