import logging
from pathlib import Path

LOG_FOLDER = "logs/"
LOG_LEVEL = logging.INFO

def setup_logger(current_time):
    # Set up logging and create a logger file
    log_file = Path(LOG_FOLDER) / f'log_file_{current_time}.log'
    logging.basicConfig(
        level=LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='w'
    )

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add console handler to the root logger
    logging.getLogger('').addHandler(console_handler)

def get_logger(name):
    return logging.getLogger(name)