import time

import numpy as np
import tensorflow as tf
import tensorflow_transform as tft

from src.context import context
from src.context import log
from src.algorithm.algorithm_v1.config import cfg
from src.algorithm.algorithm_v1.model import Model


class Trainer:
    def __init__(self):
        self.model = Model()

    def train(self):
        # the graph preprocessed by TFT preprocessing
        tf_transform_output = tft.TFTransformOutput(cfg.TARGET_DIR)

        model_fn = self.model.make_model_fn(tf_transform_output)

        run_config = tf.estimator.RunConfig(
            keep_checkpoint_max=1,
            # Checkpoints are already saved at each eval step.
            save_checkpoints_secs=1000000000,
            log_step_count_steps=1000,
            save_summary_steps=1000,
        )

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

        estimator = tf.estimator.LinearClassifier(
            feature_columns=self.model.create_feature_columns(tf_transform_output))
        estimator.train(train_input_fn, steps=2000)
        eval_evalset_result = estimator.evaluate(eval_input_fn, steps=1000, name='eval')
        print eval_evalset_result

        estimator.export_savedmodel(cfg.TARGET_DIR, make_serving_input_fn, strip_default_attrs=True)


if __name__ == "__main__":
    _start = time.time()

    trainer = Trainer()
    trainer.train()
    # trainer.eval_predict()

    context.logger.info("[total] use {}s to train".format(time.time() - _start))
