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
        self.seq_features['seq_close_price'] = ['close_b19', 'close_b18', 'close_b17', 'close_b16', 'close_b15',
                                                'close_b14', 'close_b13', 'close_b12',
                                                'close_b11', 'close_b10', 'close_b9', 'close_b8', 'close_b7',
                                                'close_b6',
                                                'close_b5', 'close_b4', 'close_b3', 'close_b2', 'close_b1', 'close_b0']

        self.simple_features = []

        self.target = "target_trend"
