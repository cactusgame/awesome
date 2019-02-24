import time
import logging

import numpy as np
import tensorflow as tf
import tensorflow_transform as tft

from src.context import context
from src.training.model import Model
from src.training.train_config import *


class Trainer:
    def __init__(self):
        self.logger = context.tflogger
        self.model = Model()

    def train(self):
        self.logger.info("start to train")
        # number of train/eval samples in tfrecord, but not the original file, because some of them of invalid
        # num_train_samples,        num_eval_samples  are init in train_config.py

        # the graph preprocessed by TFT preprocessing
        tf_transform_output = tft.TFTransformOutput(TARGET_DIR)

        model_fn = self.model.make_model_fn(tf_transform_output)

        run_config = tf.estimator.RunConfig(
            keep_checkpoint_max=1,
            # Checkpoints are already saved at each eval step.
            save_checkpoints_secs=1000000000,
            log_step_count_steps=1000,
            save_summary_steps=1000,
        )

        estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=TARGET_DIR, config=run_config)

        # Generate all `input_fn`s for the tf estimator
        train_input_fn = self.model.make_training_input_fn(
            tf_transform_output,
            train_tfrecord_fname_out + '*',
            TRAIN_BATCH_SIZE)

        eval_train_input_fn = self.model.make_training_input_fn(
            tf_transform_output,
            train_tfrecord_fname_out + '*',
            EVAL_BATCH_SIZE)

        eval_input_fn = self.model.make_training_input_fn(
            tf_transform_output,
            eval_tfrecord_fname_out + '*',
            EVAL_BATCH_SIZE)

        analysis_input_fn = self.model.make_analysis_input_fn(
            TARGET_DIR)

        # analysis_input_fn = self.model.make_analysis_input_fn(
        #     TARGET_DIR)

        # Classification metrics
        classif_auroc = [self.model.MetricKeys.Q_AUROC.format(key) for key in self.model.CLASSIF_TARGETS]
        classif_prauc = [self.model.MetricKeys.Q_PRAUC.format(key) for key in self.model.CLASSIF_TARGETS]
        classif_metric_keys = classif_auroc + classif_prauc

        worse_counter = {k: 0 for k in classif_metric_keys}
        best_metrics = {k: -np.inf for k in classif_metric_keys}

        # Save init model checkpoint and eval. We train with `steps=1` to retrieve an initial checkpoint.
        # This checkpoint is then loaded for evaluation purposes.
        estimator.train(input_fn=train_input_fn, steps=1)

        # Evaluates until: - steps batches are processed, or - input_fn raises an end-of-input exception
        estimator.evaluate(
            input_fn=eval_train_input_fn,
            steps=np.maximum(num_train_samples / EVAL_BATCH_SIZE / EVAL_TRAIN_SUBSAMPLE_FACTOR, 1),
            name='train')
        estimator.evaluate(
            input_fn=eval_input_fn, steps=np.maximum(num_eval_samples / EVAL_BATCH_SIZE, 1), name='eval')

        # This is the actual training loop. One training step is a gradient step over a minibatch.
        # 'steps' works incrementally. If you call two times train(steps=10) then training occurs in total 20 steps
        train_steps = 1
        while train_steps < TRAIN_MAX_STEPS:

            # Build the `model_fn` that builds the tf graph for train/eval/serving mode.
            model_fn = self.model.make_model_fn(tf_transform_output)

            estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=TARGET_DIR, config=run_config)

            self.logger.info('Training step {:,d}/{:,d}.'.format(train_steps, TRAIN_MAX_STEPS))

            # Train for 1 evaluation round.
            train_step = min(TRAIN_MAX_STEPS - train_steps, TRAIN_INC_STEPS)
            self.logger.info('estimator train_step {}'.format(train_step))
            estimator.train(
                input_fn=train_input_fn,
                steps=train_step)

            # Evaluation on the training set to get accurate performance metric values.
            eval_train_step = num_train_samples / EVAL_BATCH_SIZE / EVAL_TRAIN_SUBSAMPLE_FACTOR
            self.logger.info('estimator eval_train_step {}'.format(eval_train_step))
            estimator.evaluate(
                input_fn=eval_train_input_fn,
                steps=eval_train_step, name='train')

            # Evaluation on the eval set.
            eval_eval_step = num_eval_samples / EVAL_BATCH_SIZE
            self.logger.info('estimator eval_eval_step {}'.format(eval_eval_step))
            eval_eval_metrics = estimator.evaluate(
                input_fn=eval_input_fn, steps=eval_eval_step, name='eval')

            for key, value in eval_eval_metrics.iteritems():
                self.logger.info("======>" + key + " : " + str(value))

            # We keep training until all metrics worse counters exceed `p.STOP_AFTER_WORSE_EVALS_NUM`.
            # assumeption: the metrics we get should become better and better
            min_worse_counter = np.inf
            for k in classif_metric_keys:
                if eval_eval_metrics[k] > best_metrics[k]:
                    best_metrics[k] = eval_eval_metrics[k]
                    worse_counter[k] = 0
                else:
                    worse_counter[k] += 1
                min_worse_counter = min(worse_counter[k], min_worse_counter)
            if min_worse_counter > STOP_AFTER_WORSE_EVALS_NUM:
                self.logger.info('Early stopping condition reached: {}'.format(worse_counter))
                break
            self.logger.info('Best eval metrics: {}'.format(best_metrics))
            self.logger.info('Worse eval metric counter: {}'.format(worse_counter))

            train_steps += TRAIN_INC_STEPS

        # Export serving and analysis model
        # estimator.export(serving_export_dir, serving_input_fn)
        # estimator.export(ANALYSIS_DIR, analysis_input_fn)
        self.model.print_model_model_params(TARGET_DIR)

        # estimator.export_savedmodel(TARGET_DIR, analysis_input_fn,
        #                             strip_default_attrs=True)

    def _parse_line(self,line):
        CSV_TYPES = [[''], [''], [''], [''], [''],[0.0],[0.0],[0.0],[0.0],[0.0]]
        CSV_COLUMN_NAMES = ['time', 'share_id',
                            'close_b0', 'close_b1', 'close_b2','ror_05_days','ror_10_days','ror_20_days','ror_40_days','ror_60_days']

        # Decode the line into its fields
        fields = tf.decode_csv(line,record_defaults=CSV_TYPES)

        # Pack the result into a dictionary
        features = dict(zip(CSV_COLUMN_NAMES, fields))

        # Separate the label from the features
        features.pop('time')
        features.pop('ror_05_days')
        features.pop('ror_10_days')
        label = features.pop('ror_20_days')
        features.pop('ror_40_days')
        features.pop('ror_60_days')

        return features, label

    def csv_input_fn(self,csv_path, batch_size):
        # Create a dataset containing the text lines.
        dataset = tf.data.TextLineDataset(csv_path).skip(1)

        # Parse each line.
        dataset = dataset.map(self._parse_line)

        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)

        # Return the dataset.
        return dataset

    # def eval_predict(self):
    #     self.logger.info("start to train")
    #     # number of train/eval samples in tfrecord, but not the original file, because some of them of invalid
    #     # num_train_samples,        num_eval_samples  are init in train_config.py
    #
    #     # the graph preprocessed by TFT preprocessing
    #     tf_transform_output = tft.TFTransformOutput(TARGET_DIR)
    #
    #     model_fn = self.model.make_model_fn(tf_transform_output)
    #
    #     run_config = tf.estimator.RunConfig(
    #         keep_checkpoint_max=1,
    #         # Checkpoints are already saved at each eval step.
    #         save_checkpoints_secs=1000000000,
    #         log_step_count_steps=1000,
    #         save_summary_steps=1000,
    #     )
    #
    #     estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=TARGET_DIR, config=run_config)
    #
    #     vs_input_fn = self.model.make_data_validation_input_fn(
    #         tf_transform_output,
    #         eval_tfrecord_fname_out + '*',
    #         # "data/features.csv.eval.tfrecord-00000-of-00002",
    #         1)
    #
    #     self.logger.info("=====>>>>")
    #     # predictions_iter = estimator.predict(input_fn=self.csv_input_fn("data/features.csv.eval.shard-00000-of-00002",8))
    #
    #     predictions_iter = estimator.predict(input_fn=vs_input_fn)
    #
    #     i = 0
    #     for pred_dict in predictions_iter:
    #         self.logger.info(pred_dict["predictions"])
    #         i = i + 1
    #         if i>2000:
    #             break

        # self.logger.info(len(list(predictions_iter)))



    # def predict(self):
    #     # the graph preprocessed by TFT preprocessing
    #     tf_transform_output = tft.TFTransformOutput(TARGET_DIR)
    #
    #     model_fn = self.model.make_model_fn(tf_transform_output)
    #
    #     eval_input_fn = self.model.make_training_input_fn(
    #         tf_transform_output,
    #         eval_tfrecord_fname_out + '*',
    #         EVAL_BATCH_SIZE)
    #
    #     # Rebuild the input pipeline
    #     features, labels = eval_input_fn()
    #
    #     # Rebuild the model
    #     predictions = model_fn(features, labels, tf.estimator.ModeKeys.EVAL).predictions
    #
    #     # Manually load the latest checkpoint
    #     with tf.Session() as sess:
    #         checkpoint = tf.train.get_checkpoint_state(TARGET_DIR)
    #         saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta') # ....
    #         saver.restore(sess, checkpoint.model_checkpoint_path)
    #
    #         # Loop through the batches and store predictions and labels
    #         prediction_values = []
    #         label_values = []
    #         while True:
    #             try:
    #                 preds, lbls = sess.run([predictions, labels])
    #                 prediction_values += preds
    #                 label_values += lbls
    #             except tf.errors.OutOfRangeError:
    #                 break
    #         # store prediction_values and label_values somewhere


if __name__ == "__main__":
    _start = time.time()

    trainer = Trainer()
    trainer.train()
    # trainer.eval_predict()

    context.logger.info("[total] use {}s to train".format(time.time() - _start))
