# Challenge 1 - Refactor DEV code

First, the code from 'lab' folder has been refactored to be used as a module in 'solution' folder. The ojective was to create a modular code that can be easily reused and extended.

The main.py file is the entry point of the program. It loads the source data, processes it, trains a machine learning model, evaluates the model, and saves the trained model. The different variables are stored in the config folder, so it is easy to change them. There are two ways to run the code:

1. If run from the command line:
```
SRC_PATH='/path/to/source' python3 main_train.py
```

2. If run with Docker:
```
docker run -e SRC_PATH=<src_path> -v $(pwd):/app -it <image_name>
```

SRC_PATH is the path to the source data. That parameter is optional. If not provided, the default path is used: "data/raw/listings.csv". SRC_PATH has been set to be an environment variable, so it can be easily changed by the user, in case the user wants to use a different source data.

On the other hand, -v $(pwd):/app is used to mount the current directory to the container, so the logs are saved in the host machine. Also, if we want to add new data, we just need to add it to the host machine, and it will be automatically used by the container.

Some improvements in the code are:
- Classified the code into different classes and files, so it is easier to understand and maintain.
- Added error handling to the code, so it is easier to debug in case of an error.
- Added logging to the code, so it is easier to understand what the code is doing. This solution is temporary, because each time the code is run, a new log file is created. Then, it needs to be added a way to rotate the log files, so the log files are not too big. Nevertheless, it's a good way of knowing what happened in the different executions.
- Added description to the functions, classes and methods.
- Added type hints to the code.
- Added comments to the code.
- Added Unit Tests to the code. Tests are implemented for the classes and methods in src/ folder.


# Challenge 2 - Build an API

main_api.py is the entry point of the API. It defines the API and the different endpoints. It uses one of the trained models based on the input model_path parameter. The data is inputed by the user in JSON format. The API returns the id of the listing and the predicted price category.

In order to run the API, the following command can be used:
```
uvicorn main_api:app --reload
```
Then, the API can be accessed at http://127.0.0.1:8000/docs. There, the user can add the input data in JSON format and get the predicted price category.


# Challenge 3 - Dockerize your solution

2 Dockerfiles are created: one for the API and one for the training code. The Dockerfile.predict is the one that is used to run the API. The Dockerfile.train is the one that is used to train the model. 
The commands to build the images are:
```
docker build -t predict_classifier -f Dockerfile.predict .
```
```
docker build -t train_classifier -f Dockerfile.train .
```
To run the API:
```
docker run -p 8000:8000 -it predict_classifier
```
API will be accessible at http://localhost:8000/docs.

# Code structure
The code structure is the following:
project_root/
│
├── config/
│   ├── classifier_config.py
│   └── preprocessing_config.py
│
├── src/
│   ├── data_preparation.py
│   ├── data_preprocessor.py
│   ├── model_evaluator.py
│   ├── model_handler.py
│   └── setup_logger.py
│
│── main_api.py
│── main_train.py
│
├── tests/
│   ├── test_data_preparation.py
│   ├── test_model_evaluator.py
│   ├── test_model_handler.py
│   ├── test_preprocessor.py
│   └── test_setup_logger.py
│
├── Dockerfile.predict
└── Dockerfile.train