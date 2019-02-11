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
            estimator.train(
                input_fn=train_input_fn,
                steps=min(TRAIN_MAX_STEPS - train_steps, TRAIN_INC_STEPS))

            # Evaluation on the training set to get accurate performance metric values.
            estimator.evaluate(
                input_fn=eval_train_input_fn,
                steps=num_train_samples / EVAL_BATCH_SIZE / EVAL_TRAIN_SUBSAMPLE_FACTOR, name='train')

            # Evaluation on the eval set.
            eval_eval_metrics = estimator.evaluate(
                input_fn=eval_input_fn, steps=num_eval_samples / EVAL_BATCH_SIZE, name='eval')

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

        # Export the feature and wrapped_feature lists.
        # export_feature_list(serving_export_dir, data_formatter)
        #
        # # Export current commit
        # export_code_commit(TARGET_DIR)


if __name__ == "__main__":
    _start = time.time()

    trainer = Trainer()
    trainer.train()

    context.logger.info("[total] use {}s to train".format(time.time() - _start))
