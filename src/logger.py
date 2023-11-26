import logging
import os
from datetime import datetime

time_formatted = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
LOG_FILE = f"{time_formatted}.log"
logs_path = os.path.join(os.getcwd(), 'logs', LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH, # where to save logs
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

if __name__ == '__main__':
    logging.info("Logging has started.")