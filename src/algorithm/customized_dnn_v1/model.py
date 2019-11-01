import tensorflow as tf
import multiprocessing as mp

from src.base.config import cfg
from data_formatter import DataFormatter
from src.extract.feature_definition import feature_extractor_definition
from src.extract.feature_definition import feature_definition_config
from src.extract.feature_definition import TYPE_INFER
from src.context import log


class Model:
    def __init__(self):
        self.data_formatter = DataFormatter()

    def __make_target(self, transformed_features):
        """ Target definition """
        return transformed_features[self.data_formatter.target_trend]

    def make_model_fn(self):

        def model_fn(features, labels, mode):
            feature_columns = self.create_feature_columns()

            fc_close = [e for e in feature_columns if ("close_" in e.name)]
            fc_open = [e for e in feature_columns if ("open_" in e.name)]
            fc_high = [e for e in feature_columns if ("high_" in e.name)]
            fc_low = [e for e in feature_columns if ("low_" in e.name)]

            # fc_volume = [e for e in feature_columns if ("volume_" in e.name)]
            # fc_other = [e for e in feature_columns if ("volume_" not in e.name and "close_" not in e.name)]

            input_layer_close = tf.feature_column.input_layer(features=features, feature_columns=fc_close)
            input_layer_open = tf.feature_column.input_layer(features=features, feature_columns=fc_open)
            input_layer_high = tf.feature_column.input_layer(features=features, feature_columns=fc_high)
            input_layer_low = tf.feature_column.input_layer(features=features, feature_columns=fc_low)

            # images = tf.feature_column.input_layer(
            #     features=features, feature_columns=feature_columns)
            #
            # images = tf.reshape(
            #     images, shape=(-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))

            input_layer = tf.concat([input_layer_close, input_layer_open, input_layer_high, input_layer_low], axis=1)
            d = feature_definition_config['hloc_seq_step'] - 1
            input_layer = tf.reshape(input_layer, shape=(-1, d, 4))
            # --------------------------------------
            # Network definition: shared dense stack
            # --------------------------------------
            # 1st Convolutional Layer
            conv1 = tf.layers.conv1d(
                inputs=input_layer, filters=12, kernel_size=4, padding='same', activation=tf.nn.relu, name='conv1')
            pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=1, name='pool1')
            # norm1 = tf.nn.lrn(pool1, 3, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

            # 2nd Convolutional Layer
            conv2 = tf.layers.conv1d(
                inputs=pool1, filters=36, kernel_size=2, padding='same',
                activation=tf.nn.relu, name='conv2')
            # norm2 = tf.nn.lrn(conv2, 3, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
            pool2 = tf.layers.max_pooling1d(
                inputs=conv2, pool_size=3, strides=1, name='pool2')

            # Flatten Layer
            shape = pool2.get_shape()
            pool2_ = tf.reshape(pool2, [-1, shape[1] * shape[2]])

            # 1st Fully Connected Layer
            dense1 = tf.layers.dense(
                inputs=pool2_, units=512, activation=tf.nn.relu, name='dense1')

            # 2nd Fully Connected Layer
            # dense2 = tf.layers.dense(
            #     inputs=dense1, units=128, activation=tf.nn.relu, name='dense2')

            # 3rd Fully Connected Layer (Logits)
            logits = tf.layers.dense(
                inputs=dense1, units=2, activation=tf.nn.sigmoid, name='logits')

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

    def make_training_input_fn(self, files_name_pattern, batch_size):
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

            # """ Estimator input function for train/eval """
            # dataset = tf.contrib.data.make_batched_features_dataset(
            #     file_pattern=transformed_examples,
            #     # Minibatch size, generated on each `.get_next()` call.
            #     batch_size=batch_size,
            #     # features=tf_transform_output.transformed_feature_spec(),
            #     reader=tf.data.TFRecordDataset,
            #     reader_num_threads=mp.cpu_count(),
            #     parser_num_threads=mp.cpu_count(),
            #     prefetch_buffer_size=1,
            #     # we use `step` in Estimator API to controll when to stop the training
            #     # num_epochs=1,
            #     shuffle_buffer_size=cfg.SHUFFLE_BUFFER_SIZE,
            #     shuffle=True)
            # dataset = dataset.map(parse_function, num_parallel_calls=mp.cpu_count())
            #
            # iterator = dataset.make_one_shot_iterator()
            # features, labels = iterator.get_next()

            return features, target

        return input_fn

    def make_serving_input_fn(self):
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
                placeholder = tf.placeholder(shape=[None], dtype=data_formatter.get_tf_dtype(key))
                inputs[key] = placeholder
                inputs_ext[key] = placeholder

            # transformed_features = tf_transform_output.transform_raw_features(inputs)
            return tf.estimator.export.ServingInputReceiver(inputs, inputs_ext)

        return serving_input_fn

    def create_feature_columns(self):
        """
        Returns feature columns to be used by the model
        """
        # Define the feature columns: this is how the transformed inputs are
        # used by the tf model, the output of the feature columns will be
        # stacked into the `tf.feature_column.input_layer`. This object can
        # then be used by the downstream tf graph (e.g., network layers).
        base_features_columns = []
        # for key in self.data_formatter.vocabulary_features:
        #     fc = tf.feature_column.indicator_column(
        #         tf.feature_column.categorical_column_with_vocabulary_file(
        #             key=key,
        #             num_oov_buckets=1,
        #             vocabulary_file=tf_transform_output.vocabulary_file_by_name(vocab_filename=key)))
        #     base_features_columns.append(fc)

        # NUM_INT and NUM_FLOAT, already converted to numeric value by tft and scaled
        base_features_columns += [tf.feature_column.numeric_column(key, dtype=tf.float32, default_value=0.) for key in
                                  self.data_formatter.number_features]
        base_features_columns += [tf.feature_column.numeric_column(key, dtype=tf.int64, default_value=0) for key in
                                  self.data_formatter.integer_features]

        log.info('len of features_columns: {}'.format(len(base_features_columns)))
        for fc in base_features_columns:
            log.info('feature column {}'.format(fc.name))
        return base_features_columns
