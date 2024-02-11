import logging
import os
import sys
from datetime import datetime

logging_str = "[%(asctime)s: %(levelname)s: %(module)s.py: %(message)s]"
log_dir = "./logs"
log_file_name = datetime.now().strftime("%d_%h_%Y_%H%M%S")
log_filepath = os.path.join(log_dir, f"{log_file_name}.log")


logging.basicConfig(
    format = logging_str,
    handlers = [
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("ImageClassification")
logger.setLevel(logging.INFO)