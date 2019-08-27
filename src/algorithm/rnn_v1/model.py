import tensorflow as tf
import multiprocessing as mp

from src.base.config import cfg
from data_formatter import DataFormatter
from data_formatter import Target
from src.context import log
import tensorflow.contrib.rnn as rnn


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

            # feature_columns = self.create_feature_columns(tf_transform_output)
            # ????? how to generate from N input to 1 output
            inputs = tf.split(features["closes"], 21, 1)
            # 1. configure the RNN
            lstm_cell = rnn.BasicLSTMCell(
                num_units=4,
                forget_bias=1.0,
                activation=tf.nn.tanh
            )
            outputs, _ = rnn.static_rnn(cell=lstm_cell, inputs=inputs, dtype=tf.float32)

            # slice to keep only the last cell of the RNN
            outputs = outputs[-1]

            predictions = tf.layers.dense(inputs=outputs, units=1, activation=None)

            predict_output = {'values': predictions}

            if mode == tf.estimator.ModeKeys.PREDICT:
                export_outputs = {
                    'predictions': tf.estimator.export.PredictOutput(predict_output)
                }

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predict_output,
                    export_outputs=export_outputs)

                # Calculate loss using mean squared error
            loss = tf.losses.mean_squared_error(labels, predictions)

            # Create Optimiser
            optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

            # Create training operation
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())

            accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions, name='acc_op')
            tf.summary.scalar('accuracy', accuracy[1])  # for training

            # Calculate root mean squared error as additional eval metric
            eval_metric_ops = {
                # "rmse": tf.metrics.root_mean_squared_error(labels, predictions),
                # "mae": tf.metrics.mean_absolute_error(labels, predictions),
                "accuracy": accuracy
            }

            # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
            estimator_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                        loss=loss,
                                                        train_op=train_op,
                                                        eval_metric_ops=eval_metric_ops)
            return estimator_spec

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
            # todo: try RNN List features
            tensors = []
            tensors.append(transformed_features["close_b20"])
            tensors.append(transformed_features["close_b19"])
            tensors.append(transformed_features["close_b18"])
            tensors.append(transformed_features["close_b17"])
            tensors.append(transformed_features["close_b16"])
            tensors.append(transformed_features["close_b15"])
            tensors.append(transformed_features["close_b14"])
            tensors.append(transformed_features["close_b13"])
            tensors.append(transformed_features["close_b12"])
            tensors.append(transformed_features["close_b11"])
            tensors.append(transformed_features["close_b10"])
            tensors.append(transformed_features["close_b9"])
            tensors.append(transformed_features["close_b8"])
            tensors.append(transformed_features["close_b7"])
            tensors.append(transformed_features["close_b6"])
            tensors.append(transformed_features["close_b5"])
            tensors.append(transformed_features["close_b4"])
            tensors.append(transformed_features["close_b3"])
            tensors.append(transformed_features["close_b2"])
            tensors.append(transformed_features["close_b1"])
            tensors.append(transformed_features["close_b0"])

            tensors_concat = tf.stack(tensors, axis=1)
            target = tf.expand_dims(transformed_target, -1)

            # features (32,4)
            # target (32,1)
            return {"closes": tensors_concat}, target

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

            # todo: try RNN List features
            tensors = []
            tensors.append(transformed_features["close_b20"])
            tensors.append(transformed_features["close_b19"])
            tensors.append(transformed_features["close_b18"])
            tensors.append(transformed_features["close_b17"])
            tensors.append(transformed_features["close_b16"])
            tensors.append(transformed_features["close_b15"])
            tensors.append(transformed_features["close_b14"])
            tensors.append(transformed_features["close_b13"])
            tensors.append(transformed_features["close_b12"])
            tensors.append(transformed_features["close_b11"])
            tensors.append(transformed_features["close_b10"])
            tensors.append(transformed_features["close_b9"])
            tensors.append(transformed_features["close_b8"])
            tensors.append(transformed_features["close_b7"])
            tensors.append(transformed_features["close_b6"])
            tensors.append(transformed_features["close_b5"])
            tensors.append(transformed_features["close_b4"])
            tensors.append(transformed_features["close_b3"])
            tensors.append(transformed_features["close_b2"])
            tensors.append(transformed_features["close_b1"])
            tensors.append(transformed_features["close_b0"])

            tensors_concat = tf.stack(tensors, axis=1)
            return tf.estimator.export.ServingInputReceiver({"closes": tensors_concat}, inputs_ext)

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
