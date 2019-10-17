import os

from src.base.config import cfg


class Env:
    """
    initialize the env of the training
    """

    @staticmethod
    def init():
        os.system("rm -rf {}".format(cfg.TARGET_DIR))
