import tushare as ts
import logging

from src.config.app_config import app_config


class AwesomeContext:
    def __init__(self):
        ts.set_token(app_config.tushare_token)
        self.tushare = ts.pro_api()


context = AwesomeContext()

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
log = logging.getLogger("awesome")
log.setLevel(logging.INFO)

fileHandler = logging.FileHandler("temp.log")
fileHandler.setFormatter(logFormatter)
log.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
log.addHandler(consoleHandler)

tflog = logging.getLogger('tensorflow')
tflog.propagate = False
tflog.setLevel(logging.INFO)
