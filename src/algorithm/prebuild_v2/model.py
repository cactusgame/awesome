import tensorflow as tf
import multiprocessing as mp

from src.base.config import cfg
from data_formatter import DataFormatter
from data_formatter import Target
from src.context import log


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
        base_features_columns += [tf.feature_column.numeric_column(key, default_value=0.) for key in
                                  self.data_formatter.number_features]

        log.info('len of features_columns: {}'.format(len(base_features_columns)))
        for fc in base_features_columns:
            log.info('feature column {}'.format(fc.name))
        return base_features_columns
