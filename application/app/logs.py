import os
import logging
import sys
from loguru import logger

from settings import LOG_LEVEL

log_level = LOG_LEVEL

# LOG_LEVEL = logging.getLevelName(os.environ.get("LOG_LEVEL", "DEBUG"))
# LOG_LEVEL = logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO"))
LOG_LEVEL = logging.getLevelName(log_level)
JSON_LOGS = True if os.environ.get("JSON_LOGS", "0") == "1" else False

class LoggerConfig():

    def __init__(self):
        self.name = "python logger"

    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


    def setup_logging(self):
        # intercept everything at the root logger
        logging.root.handlers = [self.InterceptHandler()]
        logging.root.setLevel(LOG_LEVEL)

        # remove every other logger's handlers
        # and propagate to root logger
        for name in logging.root.manager.loggerDict.keys():
            logging.getLogger(name).handlers = []
            logging.getLogger(name).propagate = True

        # configure loguru
        logger.configure(handlers=[{"sink": sys.stdout, "serialize": JSON_LOGS}])

        return logger

