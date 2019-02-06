import os
import logging
import tensorflow as tf
import numpy as np
import multiprocessing as mp

from src.extract.feature_definition import number_keys
from src.extract.feature_definition import vocabulary_keys


class Model:
    def __init__(self):
        self.logger = logging.getLogger('tensorflow')

    def make_target(self, transformed_features, data_formatter):
        """ Target/reward definition """

        transformed_target0 = tf.cast(tf.rint(transformed_features[data_formatter.KEY_ITEM_USED_BOOL]), tf.int64)
        transformed_target1 = tf.cast(tf.rint(transformed_features[data_formatter.KEY_PLAY_AGAIN]), tf.int64)

        return transformed_target0, transformed_target1

    def make_training_input_fn(self, tf_transform_output, transformed_examples, batch_size, data_formatter):
        """ Estimator input function generator that reads transformed input data.
        :param tf_transform_output: tf.Transform output graph wrapper
        :param transformed_examples: file name
        :param batch_size: batch size
        :return: Estimator input function for train/eval
        """

        def parse_function(transformed_features):
            transformed_target = self.make_target(transformed_features, data_formatter)
            stripped_transformed_features = {k: transformed_features[k] for k in transformed_features if
                                             (k not in data_formatter.TARGET_KEYS)}
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
                # Shuffle uses a running buffer of `shuffle_buffer_size`, so only items within each buffer
                # of `shuffle_buffer_size` are shuffled. Best to make sure the dataset is shuffled beforehand.
                shuffle_buffer_size=100 * 1000,
                shuffle=True)
            dataset = dataset.map(parse_function, num_parallel_calls=mp.cpu_count())

            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()

            return features, labels

        return input_fn

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
        for key in vocabulary_keys:
            fc = tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_file(
                    key=key,
                    num_oov_buckets=1,
                    vocabulary_file=tf_transform_output.vocabulary_file_by_name(vocab_filename=key)))
            base_features_columns.append(fc)

        # NUM_INT and NUM_FLOAT, already converted to numeric value by tft and scaled
        base_features_columns += [tf.feature_column.numeric_column(key, default_value=0.) for key in number_keys]

        self.logger.info('len of features_columns: {}'.format(len(base_features_columns)))
        return base_features_columns

    def make_model_fn(self, tf_transform_output):
        def model_fn(features, labels, mode):
            """
            the model_fn feeds into Estimator
            :param features:
            :param labels:
            :param mode:
            :return:
            """
            feature_columns = self.create_feature_columns(tf_transform_output)

            # Currently we don't have categorical LF, so ignore (otherwise it errors out).
            feature_columns_lf = [e for e in feature_columns if
                                  (FeatureTags.LF in e.name and
                                   data_formatter.KEY_FEAT_VAL_VERSION not in e.name)]
            feature_columns_uf_num = [
                e for e in feature_columns if
                ((FeatureTags.LF not in e.name and data_formatter.KEY_FEAT_VAL_VERSION not in e.name) and
                 isinstance(e, _NumericColumn))]
            feature_columns_uf_cat = [
                e for e in feature_columns if
                ((FeatureTags.LF not in e.name and data_formatter.KEY_FEAT_VAL_VERSION not in e.name) and
                 not isinstance(e, _NumericColumn))]

            input_layer_lf = tf.feature_column.input_layer(
                features=features, feature_columns=feature_columns_lf)
            input_layer_uf_num = tf.feature_column.input_layer(
                features=features, feature_columns=feature_columns_uf_num)
            input_layer_uf_cat = tf.feature_column.input_layer(
                features=features, feature_columns=feature_columns_uf_cat)

            if mode == tf.estimator.ModeKeys.TRAIN:
                input_layer_lf = input_layer_lf + tf.random_normal(
                    shape=tf.shape(input_layer_lf), mean=0.0, stddev=p.INPUT_NOISE_STD, dtype=tf.float32)
                input_layer_uf_num = input_layer_uf_num + tf.random_normal(
                    shape=tf.shape(input_layer_uf_num), mean=0.0, stddev=p.INPUT_NOISE_STD, dtype=tf.float32)

            input_layer_uf_num = tf.layers.dropout(
                inputs=input_layer_uf_num,
                rate=p.DROP_UF_PROB,
                training=(mode == tf.estimator.ModeKeys.TRAIN))
            input_layer_uf = tf.concat([input_layer_uf_num, input_layer_uf_cat], axis=1)

            input_layer = tf.concat([input_layer_uf, input_layer_lf, input_layer_fvv], axis=1)

            # Network structure
            # Batch norm after linear combination and before activation. Dropout after activation.
            h1 = tf.layers.Dense(
                units=p.MODEL_NUM_UNIT_SCALE * 4,
                activation=None,
                kernel_initializer=tf.glorot_normal_initializer(),
                bias_initializer=tf.zeros_initializer()
            )(input_layer)
            h1_bn = tf.layers.batch_normalization(h1, training=(mode == tf.estimator.ModeKeys.TRAIN))
            h1_act = tf.nn.relu(h1_bn)
            h1_do = tf.layers.dropout(
                inputs=h1_act,
                rate=p.DROPOUT_PROB,
                training=(mode == tf.estimator.ModeKeys.TRAIN))
            h2 = tf.layers.Dense(
                units=p.MODEL_NUM_UNIT_SCALE * 2,
                activation=None,
                kernel_initializer=tf.glorot_normal_initializer(),
                bias_initializer=tf.zeros_initializer()
            )(h1_do)
            h2_bn = tf.layers.batch_normalization(h2, training=(mode == tf.estimator.ModeKeys.TRAIN))
            h2_act = tf.nn.relu(h2_bn)
            h2_do = tf.layers.dropout(
                inputs=h2_act,
                rate=p.DROPOUT_PROB,
                training=(mode == tf.estimator.ModeKeys.TRAIN))

            # Head for item_used
            h30 = tf.layers.Dense(
                units=p.MODEL_NUM_UNIT_SCALE,
                activation=None,
                kernel_initializer=tf.glorot_normal_initializer(),
                bias_initializer=tf.zeros_initializer()
            )(h2_do)
            h3_bn0 = tf.layers.batch_normalization(h30, training=(mode == tf.estimator.ModeKeys.TRAIN))
            h3_act0 = tf.nn.relu(h3_bn0)
            h3_do0 = tf.layers.dropout(
                inputs=h3_act0,
                rate=p.DROPOUT_PROB,
                training=(mode == tf.estimator.ModeKeys.TRAIN))
            logits0 = tf.layers.Dense(
                units=2,
                activation=None,
                kernel_initializer=tf.glorot_normal_initializer(),
                bias_initializer=tf.zeros_initializer()
            )(h3_do0)
            softmax0 = tf.contrib.layers.softmax(logits0)

            # Head for play_again
            h31 = tf.layers.Dense(
                units=p.MODEL_NUM_UNIT_SCALE,
                activation=None,
                kernel_initializer=tf.glorot_normal_initializer(),
                bias_initializer=tf.zeros_initializer()
            )(h2_do)
            h3_bn1 = tf.layers.batch_normalization(h31, training=(mode == tf.estimator.ModeKeys.TRAIN))
            h3_act1 = tf.nn.relu(h3_bn1)
            h3_do1 = tf.layers.dropout(
                inputs=h3_act1,
                rate=p.DROPOUT_PROB,
                training=(mode == tf.estimator.ModeKeys.TRAIN))
            logits1 = tf.layers.Dense(
                units=2,
                activation=None,
                kernel_initializer=tf.glorot_normal_initializer(),
                bias_initializer=tf.zeros_initializer()
            )(h3_do1)
            softmax1 = tf.contrib.layers.softmax(logits1)

            # Q-values: a combination of the `item_used` Q-value (`softmax0`) and
            # the `play_again` Q-value (`softmax1`).
            # We normalize and then add both Q-value functions together.
            q_values = (tf.div(softmax0[:, 1] - tf.reduce_min(softmax0[:, 1]),
                               tf.reduce_max(softmax0[:, 1]) - tf.reduce_min(softmax0[:, 1])) +
                        tf.div(softmax1[:, 1] - tf.reduce_min(softmax1[:, 1]),
                               tf.reduce_max(softmax1[:, 1]) - tf.reduce_min(softmax1[:, 1]))) / tf.constant(2.)

            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                labels0 = labels[0]
                labels1 = labels[1]
                onehot_labels0 = tf.one_hot(labels0, depth=2)
                onehot_labels1 = tf.one_hot(labels1, depth=2)

                # `item_used` loss definition: weighting to correct for class imbalances.
                unweighted_losses0 = tf.losses.softmax_cross_entropy(
                    onehot_labels=onehot_labels0, logits=logits0, reduction=Reduction.NONE)
                class_weights0 = tf.constant([[1., 1.]])
                sample_weights0 = tf.reduce_sum(tf.multiply(onehot_labels0, class_weights0), 1)
                loss0 = tf.reduce_mean(unweighted_losses0 * sample_weights0)

                # `play_again` loss definition: weighting to correct for class imbalances.
                unweighted_losses1 = tf.losses.softmax_cross_entropy(
                    onehot_labels=onehot_labels1, logits=logits1, reduction=Reduction.NONE)
                class_weights1 = tf.constant([[1., 1.]])
                sample_weights1 = tf.reduce_sum(tf.multiply(onehot_labels1, class_weights1), 1)
                loss1 = tf.reduce_mean(unweighted_losses1 * sample_weights1)

                loss = loss0 + loss1

                # Metrics
                auroc0 = tf.metrics.auc(labels0, softmax0[:, 1], num_thresholds=10000, curve='ROC')
                prauc0 = tf.metrics.auc(labels0, softmax0[:, 1], num_thresholds=10000, curve='PR',
                                        summation_method='careful_interpolation')

                # Metrics
                auroc1 = tf.metrics.auc(labels1, softmax1[:, 1], num_thresholds=10000, curve='ROC')
                prauc1 = tf.metrics.auc(labels1, softmax1[:, 1], num_thresholds=10000, curve='PR',
                                        summation_method='careful_interpolation')

            if mode == tf.estimator.ModeKeys.TRAIN:

                # MSE loss, optimized with Adam
                optimizer = tf.train.AdamOptimizer(1e-4)

                # This is to make sure we also update the rolling mean/var for `tf.layers.batch_normalization`
                # (which is stored outside of the Estimator scope).
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

                # TensorBoard performance metrics.
                with tf.name_scope('losses'):
                    tf.summary.scalar('loss_item_used', loss0)
                    tf.summary.scalar('loss_play_again', loss1)

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

                with tf.name_scope('layer_3_item_used'):
                    weights = tf.get_default_graph().get_tensor_by_name(os.path.split(h30.name)[0] + '/kernel:0')
                    biases = tf.get_default_graph().get_tensor_by_name(os.path.split(h30.name)[0] + '/bias:0')
                    tf.summary.histogram('weights', weights)
                    tf.summary.histogram('biases', biases)
                    tf.summary.histogram('activations', h3_act0)

                with tf.name_scope('layer_3_play_again'):
                    weights = tf.get_default_graph().get_tensor_by_name(os.path.split(h31.name)[0] + '/kernel:0')
                    biases = tf.get_default_graph().get_tensor_by_name(os.path.split(h31.name)[0] + '/bias:0')
                    tf.summary.histogram('weights', weights)
                    tf.summary.histogram('biases', biases)
                    tf.summary.histogram('activations', h3_act1)

                with tf.name_scope('logits_item_used'):
                    weights = tf.get_default_graph().get_tensor_by_name(
                        os.path.split(logits0.name)[0] + '/kernel:0')
                    biases = tf.get_default_graph().get_tensor_by_name(os.path.split(logits0.name)[0] + '/bias:0')
                    tf.summary.histogram('weights', weights)
                    tf.summary.histogram('biases', biases)
                    tf.summary.histogram('activations', h3_act0)
                with tf.name_scope('q_values_item_used'):
                    tf.summary.histogram('q0', softmax0[:, 0])
                    tf.summary.histogram('q1', softmax0[:, 1])

                with tf.name_scope('logits_play_again'):
                    weights = tf.get_default_graph().get_tensor_by_name(
                        os.path.split(logits1.name)[0] + '/kernel:0')
                    biases = tf.get_default_graph().get_tensor_by_name(os.path.split(logits1.name)[0] + '/bias:0')
                    tf.summary.histogram('weights', weights)
                    tf.summary.histogram('biases', biases)
                    tf.summary.histogram('activations', h3_act1)
                with tf.name_scope('q_values_play_again'):
                    tf.summary.histogram('q0', softmax1[:, 0])
                    tf.summary.histogram('q1', softmax1[:, 1])

                # Log a few predictions.
                target_and_q0 = tf.stack([tf.cast(labels0, tf.float32), softmax0[:, 1]], axis=1)
                target_and_q1 = tf.stack([tf.cast(labels1, tf.float32), softmax1[:, 1]], axis=1)
                logging_hook = tf.train.LoggingTensorHook({
                    'target_and_q_item_used': target_and_q0[0:10, :],
                    'target_and_q_play_again': target_and_q1[0:10, :],
                }, every_n_iter=p.LOG_FREQ_STEP)

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
                        'metrics_item_used/AUC_ROC': auroc0,
                        'metrics_item_used/AUC_PR': prauc0,
                        'metrics_play_again/AUC_ROC': auroc1,
                        'metrics_play_again/AUC_PR': prauc1,
                    }, predictions={SignatureKeys.PREDICTIONS: q_values})

            elif mode == tf.estimator.ModeKeys.PREDICT:
                """
                A policy derived from the Q-value network. This epsilon-greedy policy
                computes the seeds with the `TOP_SEEDS_K` values and replaces them according to a
                `epsilon_greedy_probability` probability with a random value in [0, 1000).
                """

                rnd = tf.random_uniform(
                    shape=(p.SEEDS_K_FINAL,), minval=0, maxval=1, dtype=tf.float32)
                rnd_seed_vector = tf.random_uniform(
                    shape=(p.SEEDS_K_FINAL,), minval=0, maxval=p.SEED_LIST_LENGTH, dtype=tf.int32)
                # Indices of top `p.TOP_SEEDS_K` Q-values.
                top_q_idx = tf.nn.top_k(q_values, k=p.TOP_SEEDS_K)[1]
                sel_q_idx = tf.random_shuffle(top_q_idx)[0:p.SEEDS_K_FINAL]
                comparison = tf.less(rnd, tf.constant(p.EPSILON_GREEDY_PROBABILITY))
                # Since seeds are in [1, `p.SEEDS_K_FINAL`], we have to add 1 to the index.
                predictions = tf.where(comparison, rnd_seed_vector, sel_q_idx) + 1

                class_labels_item_used = tf.reshape(
                    tf.tile(tf.constant(['0', '1']), (tf.shape(softmax0)[0],)),
                    (tf.shape(softmax0)[0], 2))

                class_labels_play_again = tf.reshape(
                    tf.tile(tf.constant(['0', '1']), (tf.shape(softmax1)[0],)),
                    (tf.shape(softmax1)[0], 2))

                export_outputs = {
                    # Default output (used in serving-infra)
                    # * output: Seed list. Requires using `SignatureKeys.OUTPUT` dict key, since this is
                    #   used by the downstream SRS.
                    # * eps_rnd_selection: Boolean list of whether a random seed (with eps prob)
                    #   was recommend or a predicted seed.
                    # * q_values: Q-values for all `SEED_LIST_LENGTH` seeds.
                    SignatureDefs.DEFAULT: tf.estimator.export.PredictOutput(
                        {SignatureKeys.OUTPUT: predictions,
                         "eps_rnd_selection": comparison,
                         "q_values": tf.transpose(q_values)}),
                    # Analysis output
                    SignatureDefs.ANALYSIS_ITEM_USED: tf.estimator.export.ClassificationOutput(
                        scores=softmax0,
                        classes=class_labels_item_used),
                    SignatureDefs.ANALYSIS_PLAY_AGAIN: tf.estimator.export.ClassificationOutput(
                        scores=softmax1,
                        classes=class_labels_play_again),
                    SignatureDefs.ANALYSIS_Q: tf.estimator.export.RegressionOutput(
                        value=q_values)
                }

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={SignatureKeys.PREDICTIONS: q_values},
                    export_outputs=export_outputs)

        return model_fn
