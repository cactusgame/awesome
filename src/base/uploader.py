from config import cfg
from src.utils.cos_util import CosUtil
from src.utils.utils import get_model_timestamp


class Uploader:
    def __init__(self):
        pass

    @staticmethod
    def upload_model():
        if cfg.upload_model:
            CosUtil.upload_dir(cfg.TARGET_MODEL_DIR, cfg.COS_MODEL_DIR)

            Uploader.upload_training_related()
        else:
            print("doesn't need to upload model")

    @staticmethod
    def upload_training_related():
        """
        upload files generated in training, except models
        includes tfevents(for tensor board), checkpoint, TF transform related.
        :return:
        """
        ts = get_model_timestamp()
        # in the future, you can filter the model files
        CosUtil.upload_dir(cfg.TARGET_DIR, "/models_training/{}/{}".format(cfg.MODEL_NAME, ts))
