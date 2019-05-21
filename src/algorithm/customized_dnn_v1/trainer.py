import time

import numpy as np
import tensorflow as tf
import tensorflow_transform as tft

from src.context import context
from src.context import log
from src.base.config import cfg
from model import Model


class Trainer:
    def __init__(self):
        self.model = Model()

    def train(self):
        # the graph preprocessed by TFT preprocessing
        tf_transform_output = tft.TFTransformOutput(cfg.TARGET_DIR)

        # Generate all `input_fn`s for the tf estimator
        train_input_fn = self.model.make_training_input_fn(
            tf_transform_output,
            cfg.exp_log_data_file_train_tfrecord + '*',
            cfg.TRAIN_BATCH_SIZE)
        eval_input_fn = self.model.make_training_input_fn(
            tf_transform_output,
            cfg.exp_log_data_file_eval_tfrecord + '*',
            cfg.EVAL_BATCH_SIZE)

        make_serving_input_fn = self.model.make_serving_input_fn(tf_transform_output)

        run_config = tf.estimator.RunConfig().replace(
            keep_checkpoint_max=3,
            session_config=tf.ConfigProto(device_count={'GPU': 0}))

        model_params = {
            'hidden_units': [256, 128, 64],
            'n_classes': 2,
        }
        model_fn = self.model.make_model_fn(tf_transform_output, model_params)
        estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, model_dir=cfg.TARGET_DIR)
        estimator.train(train_input_fn, steps=cfg.TRAIN_MAX_STEPS)
        eval_evalset_result = estimator.evaluate(eval_input_fn, steps=cfg.EVAL_STEPS, name='eval')
        print eval_evalset_result

        estimator.export_savedmodel(cfg.TARGET_DIR, make_serving_input_fn, strip_default_attrs=True)


if __name__ == "__main__":
    _start = time.time()

    trainer = Trainer()
    trainer.train()
    # trainer.eval_predict()

    context.logger.info("[total] use {}s to train".format(time.time() - _start))
