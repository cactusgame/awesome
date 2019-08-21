import tensorflow as tf
import multiprocessing as mp

from src.base.config import cfg
from data_formatter import DataFormatter
from data_formatter import Target
from src.context import log


class Model:
    def __init__(self):
        self.data_formatter = DataFormatter()
        self.CLASSIF_TARGETS = self.data_formatter.TARGETS

    def __make_target(self, transformed_features):
        """ Target/reward definition """

        transformed_target0 = transformed_features[Target.ROR_1_DAYS_BEYOND_0_001_BOOL]
        return transformed_target0

    def make_model_fn(self, tf_transform_output):

        def model_fn(features, labels, mode):
            """DNN with three hidden layers and learning_rate=0.1."""

            feature_columns = self.create_feature_columns(tf_transform_output)

            fc_close = [e for e in feature_columns if ("close_" in e.name)]
            fc_volume = [e for e in feature_columns if ("volume_" in e.name)]
            fc_other = [e for e in feature_columns if ("volume_" not in e.name and "close_" not in e.name)]

            input_layer_close = tf.feature_column.input_layer(features=features, feature_columns=fc_close)
            input_layer_volume = tf.feature_column.input_layer(features=features, feature_columns=fc_volume)
            input_layer_other = tf.feature_column.input_layer(features=features, feature_columns=fc_other)

            input_layer = tf.concat([input_layer_close, input_layer_volume, input_layer_other], axis=1)

            # --------------------------------------
            # Network definition: shared dense stack
            # --------------------------------------
            a_h1 = tf.layers.Dense(
                name="dense{}_{}".format(1, 512),
                units=512,
                activation=None,
                kernel_initializer=tf.glorot_normal_initializer(),
                bias_initializer=tf.zeros_initializer()
            )(input_layer)
            a_h1_bn = tf.layers.batch_normalization(a_h1, training=(mode == tf.estimator.ModeKeys.TRAIN))
            a_h1_act = tf.nn.relu(a_h1_bn)
            a_h1_do = tf.layers.dropout(
                inputs=a_h1_act,
                rate=0.0,
                training=(mode == tf.estimator.ModeKeys.TRAIN))

            a_h2 = tf.layers.Dense(
                name="dense{}_{}".format(2, 64),
                units=64,
                activation=None,
                kernel_initializer=tf.glorot_normal_initializer(),
                bias_initializer=tf.zeros_initializer()
            )(a_h1_do)
            a_h2_bn = tf.layers.batch_normalization(a_h2, training=(mode == tf.estimator.ModeKeys.TRAIN))
            a_h2_act = tf.nn.relu(a_h2_bn)
            a_h2_do = tf.layers.dropout(
                inputs=a_h2_act,
                rate=0.0,
                training=(mode == tf.estimator.ModeKeys.TRAIN))

            # Compute logits (1 per class).
            logits = tf.layers.dense(a_h2_do, 2, activation=None)

            # Compute predictions.
            predicted_classes = tf.argmax(logits, 1)
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'class_ids': predicted_classes[:, tf.newaxis],
                    'probabilities': tf.nn.softmax(logits),
                    'logits': logits,
                }
                default_export_outputs = {
                    "default_signature_key": tf.estimator.export.PredictOutput({"output": predicted_classes}),
                }
                export_outputs = dict()
                export_outputs.update(default_export_outputs)

                return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

            # Compute loss.
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

            # Compute evaluation metrics.
            accuracy = tf.metrics.accuracy(labels=labels,
                                           predictions=predicted_classes,
                                           name='acc_op')
            metrics = {'accuracy': accuracy}
            tf.summary.scalar('accuracy', accuracy[1])

            # todo: add `lose`
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

            # Create training op.
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
                train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        return model_fn

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
                # num_epochs=1,
                shuffle_buffer_size=cfg.SHUFFLE_BUFFER_SIZE,
                shuffle=True)
            dataset = dataset.map(parse_function, num_parallel_calls=mp.cpu_count())

            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()

            return features, labels

        return input_fn

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
        for key in self.data_formatter.vocabulary_features:
            fc = tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_file(
                    key=key,
                    num_oov_buckets=1,
                    vocabulary_file=tf_transform_output.vocabulary_file_by_name(vocab_filename=key)))
            base_features_columns.append(fc)

        # NUM_INT and NUM_FLOAT, already converted to numeric value by tft and scaled
        base_features_columns += [tf.feature_column.numeric_column(key, dtype=tf.float32, default_value=0.) for key in
                                  self.data_formatter.number_features]
        base_features_columns += [tf.feature_column.numeric_column(key, dtype=tf.int64, default_value=0) for key in
                                  self.data_formatter.integer_features]

        log.info('len of features_columns: {}'.format(len(base_features_columns)))
        for fc in base_features_columns:
            log.info('feature column {}'.format(fc.name))
        return base_features_columns
