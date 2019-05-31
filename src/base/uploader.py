from config import cfg
from src.utils.cos_util import CosUtil


class Uploader:
    def __init__(self):
        pass

    @staticmethod
    def upload_model():
        CosUtil.upload_dir(cfg.TARGET_MODEL_DIR, cfg.COS_MODEL_DIR)
