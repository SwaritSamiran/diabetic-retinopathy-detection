# logger setup
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_level=logging.INFO):
    logger = logging.getLogger(name)
    
    # prevent multiple handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(log_level)
    
    # create logs directory
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # file handler
    log_file = log_dir / f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

