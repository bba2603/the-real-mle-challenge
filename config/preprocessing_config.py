# Paths 
PROCESSED_FOLDER = "data/processed/"

# Columns
COLUMNS = [
    'id', 'neighbourhood_group_cleansed', 'property_type', 'room_type',
    'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms',
    'beds', 'amenities', 'price'
]

RENAMED_COLUMNS = {
    'neighbourhood_group_cleansed': 'neighbourhood',
}

FEATURE_AMENITIES = [
    'tv', 'internet', 'air_conditioning', 'kitchen',
    'heating', 'wifi', 'elevator', 'breakfast'
]

TARGET_COLUMN = 'category'