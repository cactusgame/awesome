import os
import logging
import tensorflow as tf
import numpy as np
import multiprocessing as mp

from tensorflow.python.ops.losses.losses_impl import Reduction
from tensorflow.python.saved_model import signature_constants
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io

from src.context import log
from src.context import context
from src.algorithm.prebuild_v1.config import cfg
from src.algorithm.prebuild_v1.data_formatter import DataFormatter
from src.algorithm.prebuild_v1.data_formatter import Target


class SignatureKeys(object):
    """ Enum for model signature keys """

    INPUT = 'inputs'
    OUTPUT = 'outputs'
    PREDICTIONS = 'predictions'


class SignatureDefs(object):
    """ Enum for model signature defs """

    ANALYSIS_ROR_20 = 'analysis_ror_20'
    ANALYSIS_Q = 'analysis_q'
    DEFAULT = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY


class MetricKeys(object):
    Q_AUROC = 'metrics_{}/AUC_ROC'  # for example: metrics_ror_20_days_bool/Q_AUC_ROC, metrics_ror_20_days_bool/Q_AUC_PR
    Q_PRAUC = 'metrics_{}/AUC_PR'
    Q_RMSE = 'metrics_{}/RMSE'


class Model:
    def __init__(self):
        self.data_formatter = DataFormatter()
        # Classification and regresion target definition
        self.CLASSIF_TARGETS = self.data_formatter.TARGETS

    def __make_target(self, transformed_features):
        """ Target/reward definition """

        transformed_target0 = transformed_features[Target.ROR_20_DAYS_BOOL]
        return transformed_target0

    def make_training_input_fn(self, tf_transform_output, transformed_examples, batch_size):
        """ Estimator input function generator that reads transformed input data.
        :param tf_transform_output: tf.Transform output graph wrapper
        :param transformed_examples: file name
        :param batch_size: batch size
        :return: Estimator input function for train/eval
        """

        def parse_function(transformed_features):
            transformed_target = self.__make_target(transformed_features)
            stripped_transformed_features = {k: transformed_features[k] for k in transformed_features if
                                             (k in self.data_formatter.FEATURES)}
            return stripped_transformed_features, transformed_target

        def input_fn():
            """ Estimator input function for train/eval """
            dataset = tf.contrib.data.make_batched_features_dataset(
                file_pattern=transformed_examples,
                # Minibatch size, generated on each `.get_next()` call.
                batch_size=batch_size,
                features=tf_transform_output.transformed_feature_spec(),
                reader=tf.data.TFRecordDataset,
                reader_num_threads=mp.cpu_count(),
                parser_num_threads=mp.cpu_count(),
                prefetch_buffer_size=1,
                # we use `step` in Estimator API to controll when to stop the training
                num_epochs=1,
                shuffle_buffer_size=cfg.SHUFFLE_BUFFER_SIZE,
                shuffle=True)
            # todo: to see the doc about dataset how to map the original data
            dataset = dataset.map(parse_function, num_parallel_calls=mp.cpu_count())

            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()

            return features, labels

        return input_fn

    def make_analysis_input_fn(self, tf_transform_dir):
        def analysis_input_fn():
            # Get the raw feature spec for analysis
            raw_feature_spec = self.data_formatter.get_features_spec()

            serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None])

            # A tf.parse_example operator will parse raw input files according to the analysis
            # spec `raw_feature_spec`.
            features = tf.parse_example(serialized_tf_example, raw_feature_spec)

            # Now that we have our raw examples, process them through the tf-transform
            # function computed during the preprocessing step.
            _, transformed_features = (saved_transform_io.partially_apply_saved_transform(
                os.path.join(tf_transform_dir, transform_fn_io.TRANSFORM_FN_DIR), features))

            # Remove target keys from feature list
            # todo: not sure how to filter Target keys : enabled_target_keys?
            # [transformed_features.pop(key) for key in data_formatter.TARGET_KEYS]

            # Restriction by tfma: key ust be `SignatureKeys.EXAMPLES`
            receiver_tensors = {SignatureKeys.INPUT: serialized_tf_example}

            return tf.estimator.export.ServingInputReceiver(transformed_features, receiver_tensors)

        return analysis_input_fn

    def make_serving_input_fn(self, tf_transform_output):
        """ Estimator input function generator for model serving.
        :param tf_transform_output: tf.Transform graph output wrapper.
        :return: Estimator input function for serving (prediction).
        """

        data_formatter = DataFormatter()

        def serving_input_fn():
            """
            inputs : supported features
            inputs_ext: all features
            """
            inputs, inputs_ext = {}, {}

            # Used input features
            for key in data_formatter.FEATURES:
                placeholder = tf.placeholder(
                    shape=[None], dtype=data_formatter.get_tf_dtype(key))
                inputs[key] = placeholder
                inputs_ext[key] = placeholder

            transformed_features = tf_transform_output.transform_raw_features(inputs)

            return tf.estimator.export.ServingInputReceiver(transformed_features, inputs_ext)

        return serving_input_fn

    def create_feature_columns(self, tf_transform_output):
        """
        Returns feature columns to be used by the model
        """
        # Define the feature columns: this is how the transformed inputs are
        # used by the tf model, the output of the feature columns will be
        # stacked into the `tf.feature_column.input_layer`. This object can
        # then be used by the downstream tf graph (e.g., network layers).
        base_features_columns = []
        # CATEGORY columns: tft already built vocabs
        # We define 1 additional bucket for out-of-vocab (OOV) values. This
        # way, we are able to cope with OOV-values at serving time.
        for key in self.data_formatter.VOCABULARY_FEATURES:
            fc = tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_file(
                    key=key,
                    num_oov_buckets=1,
                    vocabulary_file=tf_transform_output.vocabulary_file_by_name(vocab_filename=key)))
            base_features_columns.append(fc)

        # NUM_INT and NUM_FLOAT, already converted to numeric value by tft and scaled
        base_features_columns += [tf.feature_column.numeric_column(key, default_value=0.) for key in
                                  self.data_formatter.FEATURES]

        log.info('len of features_columns: {}'.format(len(base_features_columns)))
        for fc in base_features_columns:
            log.info('feature column {}'.format(fc.name))
        return base_features_columns

    def make_model_fn(self, tf_transform_output):
        def model_fn(features, labels, mode):
            """
            the model_fn feeds into Estimator
            """
            feature_columns = self.create_feature_columns(tf_transform_output)
            input_layer = tf.feature_column.input_layer(
                features=features, feature_columns=feature_columns)

            # Network structure
            # Batch norm after linear combination and before activation. Dropout after activation.
            h1 = tf.layers.Dense(
                units=cfg.MODEL_NUM_UNIT_SCALE * 4,
                activation=None,
                kernel_initializer=tf.glorot_normal_initializer(),
                bias_initializer=tf.zeros_initializer()
            )(input_layer)
            h1_bn = tf.layers.batch_normalization(h1, training=(mode == tf.estimator.ModeKeys.TRAIN))
            h1_act = tf.nn.relu(h1_bn)
            h1_do = tf.layers.dropout(
                inputs=h1_act,
                rate=cfg.DROPOUT_PROB,
                training=(mode == tf.estimator.ModeKeys.TRAIN))

            h2 = tf.layers.Dense(
                units=cfg.MODEL_NUM_UNIT_SCALE * 2,
                activation=None,
                kernel_initializer=tf.glorot_normal_initializer(),
                bias_initializer=tf.zeros_initializer()
            )(h1_do)
            h2_bn = tf.layers.batch_normalization(h2, training=(mode == tf.estimator.ModeKeys.TRAIN))
            h2_act = tf.nn.relu(h2_bn)
            h2_do = tf.layers.dropout(
                inputs=h2_act,
                rate=cfg.DROPOUT_PROB,
                training=(mode == tf.estimator.ModeKeys.TRAIN))

            # Head for label1
            h30 = tf.layers.Dense(
                units=cfg.MODEL_NUM_UNIT_SCALE,
                activation=None,
                kernel_initializer=tf.glorot_normal_initializer(),
                bias_initializer=tf.zeros_initializer()
            )(h2_do)
            h3_bn0 = tf.layers.batch_normalization(h30, training=(mode == tf.estimator.ModeKeys.TRAIN))
            h3_act0 = tf.nn.relu(h3_bn0)
            h3_do0 = tf.layers.dropout(
                inputs=h3_act0,
                rate=cfg.DROPOUT_PROB,
                training=(mode == tf.estimator.ModeKeys.TRAIN))
            logits0 = tf.layers.Dense(
                units=2,
                activation=None,
                kernel_initializer=tf.glorot_normal_initializer(),
                bias_initializer=tf.zeros_initializer()
            )(h3_do0)
            softmax0 = tf.contrib.layers.softmax(logits0)

            q_values = tf.div(softmax0[:, 1] - tf.reduce_min(softmax0[:, 1]),
                              tf.reduce_max(softmax0[:, 1]) - tf.reduce_min(softmax0[:, 1]))

            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                labels0 = labels  # int64 Notice: use labels but not labels[0], because we only have 1 label now.
                onehot_labels0 = tf.one_hot(labels0,
                                            depth=2)  # shape(2,0) should [batch_size, num_classes]  , logit should [batch_size, num_classes]
                # logit(?,2)
                # `ror_20_days_bool` loss definition: weighting to correct for class imbalances.
                unweighted_losses0 = tf.losses.softmax_cross_entropy(
                    onehot_labels=onehot_labels0, logits=logits0, reduction=Reduction.NONE)
                class_weights0 = tf.constant([[1., 1.]])
                sample_weights0 = tf.reduce_sum(tf.multiply(onehot_labels0, class_weights0), 1)
                loss0 = tf.reduce_mean(unweighted_losses0 * sample_weights0)

                loss = loss0

                # Metrics
                auroc0 = tf.metrics.auc(labels0, softmax0[:, 1], num_thresholds=10000, curve='ROC')
                prauc0 = tf.metrics.auc(labels0, softmax0[:, 1], num_thresholds=10000, curve='PR',
                                        summation_method='careful_interpolation')

            if mode == tf.estimator.ModeKeys.TRAIN:

                # MSE loss, optimized with Adam
                optimizer = tf.train.AdamOptimizer(cfg.FIX_LEARNING_RATE)

                # This is to make sure we also update the rolling mean/var for `tf.layers.batch_normalization`
                # (which is stored outside of the Estimator scope).
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

                # TensorBoard performance metrics.
                with tf.name_scope('losses'):
                    tf.summary.scalar('loss_ror_20', loss0)

                # TensorBoard model evolution over time.
                with tf.name_scope('layer_1'):
                    weights = tf.get_default_graph().get_tensor_by_name(os.path.split(h1.name)[0] + '/kernel:0')
                    biases = tf.get_default_graph().get_tensor_by_name(os.path.split(h1.name)[0] + '/bias:0')
                    tf.summary.histogram('weights', weights)
                    tf.summary.histogram('biases', biases)
                    tf.summary.histogram('activations', h1_act)
                with tf.name_scope('layer_2'):
                    weights = tf.get_default_graph().get_tensor_by_name(os.path.split(h2.name)[0] + '/kernel:0')
                    biases = tf.get_default_graph().get_tensor_by_name(os.path.split(h2.name)[0] + '/bias:0')
                    tf.summary.histogram('weights', weights)
                    tf.summary.histogram('biases', biases)
                    tf.summary.histogram('activations', h2_act)
                with tf.name_scope('layer_3_ror_20'):
                    weights = tf.get_default_graph().get_tensor_by_name(os.path.split(h30.name)[0] + '/kernel:0')
                    biases = tf.get_default_graph().get_tensor_by_name(os.path.split(h30.name)[0] + '/bias:0')
                    tf.summary.histogram('weights', weights)
                    tf.summary.histogram('biases', biases)
                    tf.summary.histogram('activations', h3_act0)
                with tf.name_scope('logits_ror_20'):
                    weights = tf.get_default_graph().get_tensor_by_name(
                        os.path.split(logits0.name)[0] + '/kernel:0')
                    biases = tf.get_default_graph().get_tensor_by_name(os.path.split(logits0.name)[0] + '/bias:0')
                    tf.summary.histogram('weights', weights)
                    tf.summary.histogram('biases', biases)
                    tf.summary.histogram('activations', h3_act0)
                with tf.name_scope('q_values_ror_20'):
                    tf.summary.histogram('q0', softmax0[:, 0])
                    tf.summary.histogram('q1', softmax0[:, 1])

                # Log a few predictions.label0 : ror_xxx_days_bool
                # to watch the labels and softmax in training
                label_and_softmax0 = tf.stack([tf.cast(labels0, tf.float32), softmax0[:, 1]], axis=1)
                logging_hook = tf.train.LoggingTensorHook({
                    'label_and_softmax0': label_and_softmax0[0:10, :],
                # label_and_softmax0 size is batch size in train_config "TRAIN_BATCH_SIZE"
                }, every_n_iter=cfg.LOG_FREQ_STEP)

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op,
                    training_hooks=[logging_hook])

            elif mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    # These metrics are computed over the complete eval dataset.
                    eval_metric_ops={
                        'metrics_ror_20_days_bool/AUC_ROC': auroc0,
                        'metrics_ror_20_days_bool/AUC_PR': prauc0,
                    }, predictions={SignatureKeys.PREDICTIONS: q_values})

            elif mode == tf.estimator.ModeKeys.PREDICT:
                """
                A policy derived from the Q-value network. This epsilon-greedy policy
                computes the seeds with the `TOP_SEEDS_K` values and replaces them according to a
                `epsilon_greedy_probability` probability with a random value in [0, 1000).
                """

                # Indices of top `p.TOP_SEEDS_K` Q-values.
                top_q_idx = tf.nn.top_k(q_values, k=cfg.TOP_SEEDS_K)[1]
                sel_q_idx = tf.random_shuffle(top_q_idx)[0:cfg.SEEDS_K_FINAL]
                # Since seeds are in [1, `p.SEEDS_K_FINAL`], we have to add 1 to the index.
                predictions = sel_q_idx + 1

                class_labels_ror_20 = tf.reshape(
                    tf.tile(tf.constant(['0', '1']), (tf.shape(softmax0)[0],)),
                    (tf.shape(softmax0)[0], 2))

                export_outputs = {
                    # Default output (used in serving-infra)
                    # * output: Seed list. Requires using `SignatureKeys.OUTPUT` dict key, since this is
                    #   used by the downstream SRS.
                    # * eps_rnd_selection: Boolean list of whether a random seed (with eps prob)
                    #   was recommend or a predicted seed.
                    # * q_values: Q-values for all `SEED_LIST_LENGTH` seeds.
                    SignatureDefs.DEFAULT: tf.estimator.export.PredictOutput(
                        {SignatureKeys.OUTPUT: predictions,
                         "q_values": tf.transpose(q_values)}),
                    # Analysis output
                    SignatureDefs.ANALYSIS_ROR_20: tf.estimator.export.ClassificationOutput(
                        scores=softmax0,
                        classes=class_labels_ror_20),
                    SignatureDefs.ANALYSIS_Q: tf.estimator.export.RegressionOutput(
                        value=q_values)
                }

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={SignatureKeys.PREDICTIONS: q_values},
                    export_outputs=export_outputs)

        return model_fn

    def print_model_model_params(self, working_dir):
        # Import the model back with our own Session (rather than Estimator's) to read the number
        # of parameters used.
        with tf.Session() as sess:
            checkpoint = tf.train.get_checkpoint_state(working_dir)
            saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
            saver.restore(sess, checkpoint.model_checkpoint_path)
            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                log.info('var={}: shape={} num_params={}'.format(variable.name, shape, variable_parameters))
                total_parameters += variable_parameters
            log.info('total_params={}'.format(total_parameters))
