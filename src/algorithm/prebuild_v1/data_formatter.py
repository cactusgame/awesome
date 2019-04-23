import tensorflow as tf

from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

from src.extract.feature_definition import feature_extractor_definition
from src.extract.feature_definition import FORMAT_NUMBER
from src.extract.feature_definition import FORMAT_VOCABULARY
from src.context import log


class Target:
    ROR_20_DAYS_BOOL = 'ror_20_days_bool'


class DataFormatter:
    def __init__(self):
        # feature name : feature type
        self.VOCABULARY_FEATURES = []
        # self.VOCABULARY_FEATURES.append('share_id')

        self.FEATURES = []
        self.FEATURES.append('close_b0')
        self.FEATURES.append('close_b1')
        self.FEATURES.append('close_b2')

        self.FEATURES = self.FEATURES + self.VOCABULARY_FEATURES

        self.TARGETS = []
        self.TARGETS.append(Target.ROR_20_DAYS_BOOL)

        # store the features according its type
        self.number_features = []
        self.vocabulary_features = []
        for key in self.FEATURES:
            if feature_extractor_definition[key][3] == FORMAT_NUMBER:
                self.number_features.append(key)
            elif feature_extractor_definition[key][3] == FORMAT_VOCABULARY:
                self.vocabulary_features.append(key)
            else:
                raise Exception("unsupported feature types in TFT")

        # use all features in feature_column_defination
        features_spec = {}
        for key in self.FEATURES + self.TARGETS:
            if key in feature_extractor_definition:
                feature_def = feature_extractor_definition[key]
                if feature_def[2] == "tf.FixedLenFeature":
                    features_spec[key] = tf.FixedLenFeature([], feature_def[4])
                else:
                    log.error("unsupported key : " + key)
            else:
                log.error(
                    "doesn't exist key {} in feature_extractor_definition, but in features or targets".format(key))

        self.features_metadata = dataset_metadata.DatasetMetadata(dataset_schema.from_feature_spec(features_spec))
        self.feature_spec = features_spec

    def get_features_and_targets(self):
        return self.FEATURES + self.TARGETS

    def get_features_metadata(self):
        return self.features_metadata

    def get_features_spec(self):
        return self.feature_spec

    def get_tf_dtype(self, feature_name):
        """
        according feature name, return TF dtype
        """
        return feature_extractor_definition[feature_name][4]
