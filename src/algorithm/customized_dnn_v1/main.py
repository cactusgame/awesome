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
def main(algo_id=None, train_steps=None, download_feature_db=None, do_preprocessing=None):
    try:
        _start = time.time()

        cfg.load(Config(algo_id,train_steps,download_feature_db,do_preprocessing))

        cfg.cls_data_formatter = os.path.normpath(os.path.join(os.path.dirname(__file__), 'data_formatter.py'))
        cfg.cls_coder = os.path.normpath(os.path.join(os.path.dirname(__file__), 'coder.py'))

        Env.init()

        preprocessor = Preprocessor()
        preprocessor.process()

        trainer = Trainer()
        trainer.train()

        Uploader.upload_model()

        log.info("[total] use {} seconds totally".format(time.time() - _start))
    except Exception as e:
        import traceback
        log.info(traceback.format_exc())
        time.sleep(60 * 60 * 24)  # wait to check logs


if __name__ == "__main__":
    main()