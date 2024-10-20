# Challenge 1 - Refactor DEV code

First, the code from 'lab' folder has been refactored to be used as a module in 'solution' folder. The ojective was to create a modular code that can be easily reused and extended.

The main.py file is the entry point of the program. It loads the source data, processes it, trains a machine learning model, evaluates the model, and saves the trained model. The different variables are stored in the config folder, so it is easy to change them. There are two ways to run the code:

1. If run from the command line:
```
SRC_PATH='/path/to/source' python3 main.py
```

2. If run with Docker:
```
docker run -e SRC_PATH=<src_path> <image_id>
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

The code structure is the following:
project_root/
│
├── solution/
│   ├── config/
│   │   ├── classifier_config.py
│   │   └── preprocessing_config.py
│   ├── src/
│   │   ├── data_preparation.py
│   │   ├── model_evaluator.py
│   │   ├── model_handler.py
│   │   ├── preprocessor.py
│   │   └── setup_logger.py
│   └── main.py
│
├── tests/
│   ├── test_data_preparation.py
│   ├── test_model_evaluator.py
│   ├── test_model_handler.py
│   ├── test_preprocessor.py
│   └── test_setup_logger.py
│
│
└── Dockerfile

# Challenge 2 - Build an API