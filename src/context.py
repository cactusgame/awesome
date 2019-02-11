import tushare as ts
import logging

from src.config.app_config import app_config


class AwesomeContext:
    def __init__(self):
        ts.set_token(app_config.tushare_token)
        self.tushare = ts.pro_api()

        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        rootLogger = logging.getLogger("awesome")
        rootLogger.setLevel(logging.INFO)

        fileHandler = logging.FileHandler("temp.log")
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)
        self.logger = rootLogger

        tflogger = logging.getLogger('tensorflow')
        tflogger.propagate = False
        tflogger.setLevel(logging.INFO)
        self.tflogger = tflogger


context = AwesomeContext()
