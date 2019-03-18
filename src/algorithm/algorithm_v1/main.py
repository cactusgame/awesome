import time

from src.base.config import cfg
from src.utils.logger import log
from src.base.preprocess.preprocessor import Preprocessor
from src.algorithm.algorithm_v1.config import Config


def main():
    _start = time.time()

    cfg.load(Config())

    cfg.cls_data_formatter = 'src/algorithm/algorithm_v1/data_formatter.py'

    preprocessor = Preprocessor()
    preprocessor.process()

    log.info("[total] use {}s to preprocess".format(time.time() - _start))


if __name__ == "__main__":
    main()
