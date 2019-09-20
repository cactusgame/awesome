import tensorflow as tf
import multiprocessing as mp

from src.extract.feature_definition import feature_extractor_definition
from src.extract.feature_definition import TYPE_INFER

from src.base.config import cfg
from data_formatter import DataFormatter
# from src.context import log
import tensorflow.contrib.rnn as rnn

OUTPUT_SEQUENCE_LENGTH = 1

# VALUES_FEATURE_NAME = ["closes"]
# forget_bias = 1.0
# hidden_units = [64, 8]
# TARGET_LABELS = [0, 1]


class Model:
    def __init__(self):
        self.schema = DataFormatter()

    def make_model_fn(self):
        def model_fn(features, labels, mode):
            inputs = tf.split(features['seq_close_price'], len(self.schema.seq_features['seq_close_price']), 1)

            hidden_units = 64
            forget_bias = 1.0

            # 1. configure the RNN
            lstm_cell = rnn.BasicLSTMCell(
                num_units=hidden_units,
                forget_bias=forget_bias,
                activation=tf.nn.tanh
            )
            outputs, _ = rnn.static_rnn(cell=lstm_cell,
                                        inputs=inputs,
                                        dtype=tf.float32)
            # slice to keep only the last cell of the RNN
            outputs = outputs[-1]
            print('last outputs={}'.format(outputs))

            predictions = tf.layers.dense(inputs=outputs,
                                          units=OUTPUT_SEQUENCE_LENGTH,
                                          activation=None)

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

            tf.summary.scalar('loss', loss)
            # Create Optimiser
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

            # Create training operation
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

            # Calculate root mean squared error as additional eval metric
            eval_metric_ops = {
                "rmse": tf.metrics.root_mean_squared_error(labels, predictions),
                "mae": tf.metrics.mean_absolute_error(labels, predictions)
            }

            # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
            estimator_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                        loss=loss,
                                                        train_op=train_op,
                                                        eval_metric_ops=eval_metric_ops)
            return estimator_spec

        return model_fn

    def make_training_input_fn(self, files_name_pattern, batch_size):
        """ Estimator input function generator that reads transformed input data.
        :param tf_transform_output: tf.Transform output graph wrapper
        :param transformed_examples: file name
        :param batch_size: batch size
        :return: Estimator input function for train/eval
        """

        def create_seq_feature(transformed_features, feature_column_names):
            feature_list = []
            for col_name in feature_column_names:
                feature_list.append(transformed_features[col_name])

            feature_tensor = tf.convert_to_tensor(feature_list, dtype=tf.float32)
            feature_tensor = tf.transpose(feature_tensor, [1, 0])
            return feature_tensor

        def parse_function(transformed_features):
            features = {}
            for feature_name, feature_column_names in self.schema.seq_features.items():
                features[feature_name] = create_seq_feature(transformed_features, feature_column_names)

            target = transformed_features[self.schema.target]
            label = tf.expand_dims(target, -1)
            return features, label

        def input_fn():
            feature_columns = [key for key in feature_extractor_definition.keys() if
                               feature_extractor_definition[key][5] != TYPE_INFER]

            dataset = tf.contrib.data.make_csv_dataset(
                file_pattern=files_name_pattern,
                column_names=feature_columns,
                batch_size=batch_size,
                num_parallel_reads=2,
            )
            dataset = dataset.map(parse_function, num_parallel_calls=mp.cpu_count())
            # dataset = dataset.repeat(num_epochs)
            iterator = dataset.make_one_shot_iterator()

            features, target = iterator.get_next()
            return features, target

        return input_fn

    def make_serving_input_fn(self):
        """
        fake impl
        :return:
        """

        def serving_input_fn():
            """
            inputs : supported features
            inputs_ext: all features
            """
            inputs, inputs_ext = {}, {}

            # Used input features
            key = "close_price"
            placeholder = tf.placeholder(name=key, shape=[None], dtype=tf.float32)
            inputs[key] = placeholder
            inputs_ext[key] = placeholder

            return tf.estimator.export.ServingInputReceiver(inputs, inputs_ext)

        return serving_input_fn
