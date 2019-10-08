import tensorflow as tf
import multiprocessing as mp

from src.extract.feature_definition import feature_extractor_definition
from src.extract.feature_definition import TYPE_INFER

from src.base.config import cfg
from data_formatter import DataFormatter
# from src.context import log
import tensorflow.contrib.rnn as rnn

OUTPUT_SEQUENCE_LENGTH = 1
TARGET_LABELS = [0, 1]

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
            # I suppose the last output should be the predict price
            # last_output = outputs[-1]
            # print('last outputs={}'.format(last_output))

            result = outputs[-1] - outputs[-2]
            logits = tf.layers.dense(inputs=result,
                                          units=len(TARGET_LABELS),
                                          activation=tf.nn.sigmoid)

            predict_output = {'values': logits}

            if mode == tf.estimator.ModeKeys.PREDICT:
                probabilities = tf.nn.softmax(logits)
                predicted_indices = tf.argmax(probabilities, 1)

                # Convert predicted_indices back into strings
                predictions = {
                    'class': tf.gather(TARGET_LABELS, predicted_indices),
                    'probabilities': probabilities
                }
                export_outputs = {
                    'prediction': tf.estimator.export.PredictOutput(predictions)
                }

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predict_output,
                    export_outputs=export_outputs)

            # loss for classification
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels))
            # loss for regression
            # loss = tf.losses.mean_squared_error(labels, logits)
            tf.summary.scalar('loss', loss)


            # ========== train ==========
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

            # ========== eval =============
            probabilities = tf.nn.softmax(logits)
            predicted_indices = tf.argmax(probabilities, 1)
            labels_one_hot = tf.one_hot(
                labels,
                depth=len(TARGET_LABELS)
            )
            acc = tf.metrics.accuracy(labels, predicted_indices)

            # ========= train and eval ========
            tf.summary.scalar('acc', acc[1])

            eval_metric_ops = {
                # eval metrics for regression
                # "rmse": tf.metrics.root_mean_squared_error(labels, logits),
                # "mae": tf.metrics.mean_absolute_error(labels, logits)

                # eval metrics for classification
                'accuracy': acc,
                'auroc': tf.metrics.auc(labels_one_hot, probabilities)
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
            # label = tf.expand_dims(target, -1)
            return features, target

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
            key = "seq_close_price"
            placeholder = tf.placeholder(name=key, shape=[None, 20], dtype=tf.float32)
            inputs[key] = placeholder
            inputs_ext[key] = placeholder

            return tf.estimator.export.ServingInputReceiver(inputs, inputs_ext)

        return serving_input_fn
