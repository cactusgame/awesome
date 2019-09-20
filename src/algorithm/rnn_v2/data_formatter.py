import tensorflow as tf

from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

from src.extract.feature_definition import feature_extractor_definition
from src.extract.feature_definition import FORMAT_NUMBER
from src.extract.feature_definition import FORMAT_INTEGER
from src.extract.feature_definition import FORMAT_VOCABULARY
from src.context import log

from collections import OrderedDict


class DataFormatter:
    def __init__(self):
        """
        there are 2 types of features.
        - simple feature: 1 column
        - seq feature: multiple columns and has order
        """
        self.seq_features = OrderedDict()
        self.seq_features['seq_close_price'] = ['close_b2', 'close_b1', 'close_b0']

        self.simple_features = []

        self.target = "target_close_price"
