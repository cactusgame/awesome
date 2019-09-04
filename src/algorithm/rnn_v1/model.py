import tensorflow as tf
import multiprocessing as mp

from src.base.config import cfg
from data_formatter import DataFormatter
from data_formatter import Target
from src.context import log
import tensorflow.contrib.rnn as rnn

SEQUENCE_LENGTH = 21
VALUES_FEATURE_NAME = ["closes"]
forget_bias = 1.0
hidden_units = [64, 8]
TARGET_LABELS = [0, 1]


class Model:
    def __init__(self):
        self.data_formatter = DataFormatter()
        self.CLASSIF_TARGETS = self.data_formatter.TARGETS

    def __make_target(self, transformed_features):
        transformed_target0 = transformed_features[Target.ROR_1_DAYS_BEYOND_0_001_BOOL]
        return transformed_target0

    def make_model_fn(self, tf_transform_output):

        def model_fn(features, labels, mode):
            # inputs = tf.split(features["closes"], SEQUENCE_LENGTH, 1)
            inputs = features["closes"]
            inputs = tf.reshape(tensor=inputs, shape=[64,21,1])

            def _add_conv_layers(input_layer):
                dropout = 0
                batch_norm = False
                num_conv = [48, 64, 96] # "Number of conv layers along with number of filters per layer."
                conv_len = [5, 5, 3] # Length of the convolution filters.

                convolved = input_layer
                for i in range(len(num_conv)):
                    convolved_input = convolved
                    if batch_norm:
                        convolved_input = tf.layers.batch_normalization(
                            convolved_input,
                            training=(mode == tf.estimator.ModeKeys.TRAIN))
                    # Add dropout layer if enabled and not first convolution layer.
                    # if i > 0 and params.dropout:
                    #     convolved_input = tf.layers.dropout(
                    #         convolved_input,
                    #         rate=params.dropout,
                    #         training=(mode == tf.estimator.ModeKeys.TRAIN))
                    convolved = tf.layers.conv1d(
                        convolved_input,
                        filters=num_conv[i],
                        kernel_size=conv_len[i],
                        activation=None,
                        strides=1,
                        padding="same",
                        name="conv1d_%d" % i)
                return convolved

            def _add_rnn_layers(input_layer):

                input_layer = tf.split(input_layer, SEQUENCE_LENGTH, 1)
                input_layer = [tf.squeeze(l) for l in input_layer]
                rnn_layers = [tf.nn.rnn_cell.LSTMCell(
                    num_units=size,
                    forget_bias=forget_bias,
                    activation=tf.nn.tanh) for size in hidden_units]
                multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
                outputs, _ = tf.nn.static_rnn(cell=multi_rnn_cell,
                                              inputs=input_layer,
                                              dtype=tf.float32)
                outputs = outputs[-1]
                return outputs

            def _add_fc_layers(rnn_output):
                return tf.layers.dense(inputs=rnn_output, units=len(TARGET_LABELS), activation=tf.nn.tanh)

            cnn_output = _add_conv_layers(inputs)
            rnn_output = _add_rnn_layers(cnn_output)
            logits = _add_fc_layers(rnn_output)

            probabilities = tf.nn.softmax(logits)
            predicted_indices = tf.argmax(probabilities, 1)

            if mode == tf.estimator.ModeKeys.PREDICT:
                # Convert predicted_indices back into strings
                predictions = {
                    'class': tf.gather(TARGET_LABELS, predicted_indices),
                    'probabilities': probabilities
                }
                export_outputs = {
                    'prediction': tf.estimator.export.PredictOutput(predictions)
                }

                # Provide an estimator spec for `ModeKeys.PREDICT` modes.
                return tf.estimator.EstimatorSpec(mode,
                                                  predictions=predictions,
                                                  export_outputs=export_outputs)

            acc = tf.metrics.accuracy(labels, predicted_indices, name='acc_op')  # why the name acc_op?

            # Calculate loss using softmax cross entropy
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels))

            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', acc[1])

            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.AdamOptimizer()
                train_op = optimizer.minimize(
                    loss=loss, global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  train_op=train_op)

            if mode == tf.estimator.ModeKeys.EVAL:
                # Return accuracy and area under ROC curve metrics
                labels_one_hot = tf.one_hot(
                    labels,
                    depth=len(TARGET_LABELS),
                    on_value=True,
                    off_value=False,
                    dtype=tf.bool
                )
                eval_metric_ops = {
                    'accuracy': acc,
                    'auroc': tf.metrics.auc(labels_one_hot, probabilities)
                }
                return tf.estimator.EstimatorSpec(mode,
                                                  loss=loss,
                                                  eval_metric_ops=eval_metric_ops)

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
            # target (32,1) -> target(32,)
            return {"closes": tensors_concat}, tf.squeeze(target)  # target remove dim1

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
