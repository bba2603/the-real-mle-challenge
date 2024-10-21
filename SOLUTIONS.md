# Challenge 1 - Refactor DEV code

First, the code from 'lab' folder has been refactored to be used as a module in 'solution' folder. The ojective was to create a modular code that can be easily reused and extended.

The main.py file is the entry point of the program. It loads the source data, processes it, trains a machine learning model, evaluates the model, and saves the trained model. The different variables are stored in the config folder, so it is easy to change them. There are two ways to run the code:

1. If run from the command line:
```
SRC_PATH='/path/to/source' python3 main.py
```

2. If run with Docker:
```
docker run -e SRC_PATH=<src_path> -v $(pwd):/app -it <image_name>
```

SRC_PATH is the path to the source data. That parameter is optional. If not provided, the default path is used: "data/raw/listings.csv". SRC_PATH has been set to be an environment variable, so it can be easily changed by the user, in case the user wants to use a different source data.

Some improvements in the code are:
- Classified the code into different classes and files, so it is easier to understand and maintain.
- Added error handling to the code, so it is easier to debug in case of an error.
- Added logging to the code, so it is easier to understand what the code is doing. This solution is temporary, because each time the code is run, a new log file is created. Then, it needs to be added a way to rotate the log files, so the log files are not too big. Nevertheless, it's a good way of knowing what happened in the different executions.
- Added description to the functions, classes and methods.
- Added type hints to the code.
- Added comments to the code.
- Added Unit Tests to the code.

# Challenge 2 - Build an API

main_api.py is the entry point of the API. It defines the API and the different endpoints. It uses one of the trained models based on the input model_path parameter. The data is inputed by the user in JSON format. The API returns the id of the listing and the predicted price category.

# Challenge 3 - Dockerize your solution

2 Dockerfiles are created: one for the API and one for the training code. The Dockerfile.predict is the one that is used to run the API. The Dockerfile.train is the one that is used to train the model. 
The command to build the image is:
```
docker build -t <image_name> -f Dockerfile.<name> .
```
For example:
```
docker build -t predict_classifier -f Dockerfile.predict .
```
```
docker build -t train_classifier -f Dockerfile.train .
```

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
│── Dockerfile.predict
└── Dockerfile.train