import tensorflow as tf

from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

from src.extract.feature_definition import feature_extractor_definition
from src.extract.feature_definition import feature_definition_config
from src.extract.feature_definition import FORMAT_NUMBER
from src.extract.feature_definition import FORMAT_INTEGER
from src.extract.feature_definition import FORMAT_VOCABULARY
from src.context import log

from collections import OrderedDict


class DataFormatter:
    def __init__(self):
        """
        - simple feature: 1 column
        """
        self.FEATURES = []

        for i in range(feature_definition_config["hloc_seq_step"] - 1, 0, -1):
            self.FEATURES.append('close_b{}'.format(i - 1))
        for i in range(feature_definition_config["hloc_seq_step"] - 1, 0, -1):
            self.FEATURES.append('open_b{}'.format(i - 1))
        for i in range(feature_definition_config["hloc_seq_step"] - 1, 0, -1):
            self.FEATURES.append('high_b{}'.format(i - 1))
        for i in range(feature_definition_config["hloc_seq_step"] - 1, 0, -1):
            self.FEATURES.append('low_b{}'.format(i - 1))
        for i in range(feature_definition_config["hloc_seq_step"] - 1, 0, -1):
            self.FEATURES.append('vol_b{}'.format(i - 1))

        self.target_trend = "target_close_trend"
        self.TARGETS = []
        self.TARGETS.append(self.target_trend)

        # store the features by data type
        self.integer_features = []
        self.number_features = []
        self.vocabulary_features = []
        for key in self.FEATURES:
            feature_key_type = feature_extractor_definition[key][3]
            if feature_key_type == FORMAT_NUMBER:
                self.number_features.append(key)
            elif feature_key_type == FORMAT_INTEGER:
                self.integer_features.append(key)
            elif feature_key_type == FORMAT_VOCABULARY:
                self.vocabulary_features.append(key)
            else:
                raise Exception("unsupported feature types in TFT")

        # use all features in feature_column_defination
        # features_spec = {}
        # for key in self.FEATURES + self.TARGETS:
        #     if key in feature_extractor_definition:
        #         feature_def = feature_extractor_definition[key]
        #         if feature_def[2] == "tf.FixedLenFeature":
        #             features_spec[key] = tf.FixedLenFeature([], feature_def[4])
        #         else:
        #             log.error("unsupported key : " + key)
        #     else:
        #         log.error(
        #             "doesn't exist key {} in feature_extractor_definition, but in features or targets".format(key))
        #
        # self.features_metadata = dataset_metadata.DatasetMetadata(dataset_schema.from_feature_spec(features_spec))
        # self.feature_spec = features_spec

    def get_features_and_targets(self):
        return self.FEATURES + self.TARGETS

    # def get_features_metadata(self):
    #     return self.features_metadata
    #
    # def get_features_spec(self):
    #     return self.feature_spec

    def get_tf_dtype(self, feature_name):
        """
        according feature name, return TF dtype
        """
        return feature_extractor_definition[feature_name][4]
