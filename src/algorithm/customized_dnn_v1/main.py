import time
import os
import click

from src.base.config import cfg
from src.context import log
from src.base.preprocess.preprocessor import Preprocessor
from config import Config
from trainer import Trainer
from src.base.uploader import Uploader
from src.base.env import Env


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--algo_id", type=str)
@click.option("--train_steps", type=int)
@click.option("--download_feature_db", type=bool)
@click.option("--do_preprocessing", type=bool)
@click.option("--upload_model", type=bool)
def main(algo_id=None, train_steps=None, download_feature_db=None, do_preprocessing=None, upload_model=None):
    try:
        _start = time.time()

        cfg.load(Config(algo_id, train_steps, download_feature_db, do_preprocessing, upload_model))

        cfg.cls_data_formatter = os.path.normpath(os.path.join(os.path.dirname(__file__), 'data_formatter.py'))
        cfg.cls_coder = os.path.normpath(os.path.join(os.path.dirname(__file__), 'coder.py'))

        Env.init()

        preprocessor = Preprocessor()
        preprocessor.process()

        trainer = Trainer()
        trainer.train()

        Uploader.upload_model()

        # import tensorflow as tf
        # saved_model_dir = "gen/model/1559483504"
        #
        # predictor_fn = tf.contrib.predictor.from_saved_model(
        #     export_dir=saved_model_dir,
        #     signature_def_key="default_signature_key"
        # )
        # output = predictor_fn({'share_id': "000001.SZ",
        #                        "close_b20":-1.3997,
        #                        "close_b11":-1.7812,
        #                        "close_b10":-0.8656,
        #                        "close_b13":-1.3043,
        #                        "close_b12":-1.1136,
        #                        "close_b15":-1.3043,
        #                        "close_b14":-1.3043,
        #                        "close_b17":-0.7702,
        #                        "close_b16":0.4314,
        #                        "close_b19":0.6413,
        #                        "close_b18":0.832,
        #                        "close_b1":0.7557,
        #                        "close_b0":0.9655,
        #                        "close_b3":0.6603,
        #                        "close_b2":0.5268,
        #                        "close_b5":1.1181,
        #                        "close_b4":0.5459,
        #                        "close_b7":0.4696,
        #                        "close_b6":0.7939,
        #                        "close_b9":0.8702,
        #                        "close_b8":1.2326})
        # print(output)
        log.info("[total] use {} seconds totally".format(time.time() - _start))
    except Exception as e:
        import traceback
        log.info(traceback.format_exc())
        time.sleep(60 * 60 * 24)  # wait to check logs


if __name__ == "__main__":
    main()
