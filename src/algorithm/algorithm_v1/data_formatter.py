import tensorflow as tf

from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

from src.extract.feature_definition import feature_column_definition
from src.extract.feature_definition import TYPE_FEATURE
from src.extract.feature_definition import TYPE_DICT
from src.context import context


class DataFormatter:
    def __init__(self):
        self.ordered_columns = []

        RAW_DATA_FEATURE_SPEC = dict()
        # use all features in feature_column_defination
        # in the future, will use only part of the defined features
        for key, value in feature_column_definition.iteritems():
            if value[6]:
                if value[2] == "tf.FixedLenFeature":
                    RAW_DATA_FEATURE_SPEC[key] = tf.FixedLenFeature([], value[4])
                else:
                    context.logger.error("unsupport key : " + key)

        self.RAW_DATA_METADATA = dataset_metadata.DatasetMetadata(
            dataset_schema.from_feature_spec(RAW_DATA_FEATURE_SPEC))

        # feature name : feature type
        self.USED_FEATURES = {}
        for key, value in feature_column_definition.iteritems():
            if value[5] == TYPE_FEATURE:
                self.USED_FEATURES[key] = value[0]  # python type



    def init_columns(self, columns_str):
        """
        init all columns name by a fix order.
        Also, the columns_str is the header of the csv file
        :param columns_str:
        :return:
        """
        self.ordered_columns = columns_str.strip().split(',')

    def get_ordered_columns(self):
        return self.ordered_columns

    def get_raw_data_metadata(self):
        return self.RAW_DATA_METADATA

    def get_tf_dtype(self, feature_name):
        """
        according feature name, return TF dtype
        :param feature_name:
        :return:
        """
        return TYPE_DICT[self.USED_FEATURES[feature_name]]
