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

        # Generate all `input_fn`s for the tf estimator
        train_input_fn = self.model.make_training_input_fn(
            cfg.get_shard_file("feature_train") + '*',
            cfg.TRAIN_BATCH_SIZE)
        eval_input_fn = self.model.make_training_input_fn(
            cfg.get_shard_file("feature_eval") + '*',
            cfg.EVAL_BATCH_SIZE)

        make_serving_input_fn = self.model.make_serving_input_fn()

        run_config = tf.estimator.RunConfig().replace(
            save_checkpoints_secs=cfg.SAVE_MODEL_SECONDS,
            keep_checkpoint_max=3,
            session_config=tf.ConfigProto(device_count={'GPU': 0}))

        model_fn = self.model.make_model_fn()
        estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, model_dir=cfg.TARGET_DIR)

        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=cfg.EVAL_STEPS,
                                          name='evaluation', start_delay_secs=5, throttle_secs=cfg.EVAL_SECONDS)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=cfg.TRAIN_MAX_STEPS)

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

        # estimator.export_savedmodel(cfg.TARGET_MODEL_DIR, make_serving_input_fn, strip_default_attrs=True)


if __name__ == "__main__":
    _start = time.time()

    trainer = Trainer()
    trainer.train()
    # trainer.eval_predict()

    context.logger.info("[total] use {}s to train".format(time.time() - _start))
