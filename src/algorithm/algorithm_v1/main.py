import time
import os

from src.base.config import cfg
from src.context import log
from src.base.preprocess.preprocessor import Preprocessor
from src.algorithm.algorithm_v1.config import Config
from src.algorithm.algorithm_v1.trainer import Trainer

def main():
    _start = time.time()

    cfg.load(Config())

    cfg.cls_data_formatter = os.path.normpath(os.path.join(os.path.dirname(__file__), 'data_formatter.py'))
    cfg.cls_coder = os.path.normpath(os.path.join(os.path.dirname(__file__), 'coder.py'))

    preprocessor = Preprocessor()
    preprocessor.process()

    trainer = Trainer()
    trainer.train()

    log.info("[total] use {} seconds totally".format(time.time() - _start))


if __name__ == "__main__":
    main()
