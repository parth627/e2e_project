'''
Basic logging functionality
'''
import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_path = os.path.join("../","logs",LOG_FILE)

os.makedirs(log_path,exist_ok=True) #even there is file, folder we append in it.

LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)

logging.basicConfig(filename=LOG_FILE_PATH,
                    format="[ %(asctime)s) ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
                    level=logging.INFO
                    )
