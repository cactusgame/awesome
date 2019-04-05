import time

import numpy as np
import tensorflow as tf
import tensorflow_transform as tft

from src.context import context
from src.utils.logger import log
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
        estimator.train(train_input_fn, steps=100)
        eval_evalset_result = estimator.evaluate(eval_input_fn, steps=100, name='eval')
        print eval_evalset_result

        estimator.export_savedmodel(cfg.TARGET_DIR, make_serving_input_fn, strip_default_attrs=True)

    # # todo
    # def _parse_line(self, line):
    #     CSV_TYPES = [[''], [''], [''], [''], [''], [0.0], [0.0], [0.0], [0.0], [0.0]]
    #     CSV_COLUMN_NAMES = ['time', 'share_id',
    #                         'close_b0', 'close_b1', 'close_b2', 'ror_05_days', 'ror_10_days', 'ror_20_days',
    #                         'ror_40_days', 'ror_60_days']
    #
    #     # Decode the line into its fields
    #     fields = tf.decode_csv(line, record_defaults=CSV_TYPES)
    #
    #     # Pack the result into a dictionary
    #     features = dict(zip(CSV_COLUMN_NAMES, fields))
    #
    #     # Separate the label from the features
    #     features.pop('time')
    #     features.pop('ror_05_days')
    #     features.pop('ror_10_days')
    #     label = features.pop('ror_20_days')
    #     features.pop('ror_40_days')
    #     features.pop('ror_60_days')
    #
    #     return features, label
    #
    # def csv_input_fn(self, csv_path, batch_size):
    #     # Create a dataset containing the text lines.
    #     dataset = tf.data.TextLineDataset(csv_path).skip(1)
    #
    #     # Parse each line.
    #     dataset = dataset.map(self._parse_line)
    #
    #     # Shuffle, repeat, and batch the examples.
    #     dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    #
    #     # Return the dataset.
    #     return dataset


if __name__ == "__main__":
    _start = time.time()

    trainer = Trainer()
    trainer.train()
    # trainer.eval_predict()

    context.logger.info("[total] use {}s to train".format(time.time() - _start))
