import logging
import sys
from rich.logging import RichHandler

def get_logger(name: str = "uaam") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = RichHandler(rich_tracebacks=True, markup=True)
    stream.setLevel(logging.INFO)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    return logger


## 任意入口处
#from src.utils.seed import set_global_seed
#from src.utils.logging import get_logger

#set_global_seed(3407)
#logger = get_logger(__name__)
#logger.info("Seed set. Starting training...")
