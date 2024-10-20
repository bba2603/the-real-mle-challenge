# Mappings
MAP_ROOM_TYPE = {
    "Shared room": 1, "Private room": 2, "Entire home/apt": 3, "Hotel room": 4
}
MAP_NEIGHB = {
    "Bronx": 1, "Queens": 2, "Staten Island": 3, "Brooklyn": 4, "Manhattan": 5
}

# Paths of the folders
MODEL_FOLDER = "models/"
RESULTS_FOLDER = "results/"

# Features for the model
FEATURE_NAMES = [
    'neighbourhood', 'room_type', 'accommodates', 'bathrooms', 'bedrooms'
]

# Split data
SPLIT_RATIO = 0.15
RANDOM_STATE_SPLIT = 1

# Classifier parameters
N_ESTIMATORS = 500
RANDOM_STATE_CLASSIFIER = 0
CLASS_WEIGHT = 'balanced'
N_JOBS = 4