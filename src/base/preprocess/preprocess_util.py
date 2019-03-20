import apache_beam as beam
import tensorflow_transform as tft

from src.context import context
from src.extract.feature_definition import *


class PreprocessingFunction(object):
    """
    A transformer that preprocessed data in csv into TFT format columns
    """

    def __init__(self, data_formatter):
        self.data_formatter = data_formatter

    def transform_to_tfrecord(self, inputs):
        """Preprocess raw input columns into transformed columns."""
        outputs = inputs.copy()

        for key in self.data_formatter.number_features:
            outputs[key] = tft.scale_to_z_score((outputs[key]))

        for key in self.data_formatter.vocabulary_features:
            tft.vocabulary(inputs[key], vocab_filename=key)

        return outputs


class MapAndFilterErrors(beam.PTransform):
    class _MapAndFilterErrorsDoFn(beam.DoFn):

        def __init__(self, fn):
            self.logger = context.logger
            self._fn = fn

        def process(self, element):
            try:
                yield self._fn(element)
            except Exception as e:
                self.logger.warning('Data row processing error: \"{}\". Skipping row.'.format(e))

    def __init__(self, fn):
        self._fn = fn

    def expand(self, pcoll):
        return (pcoll
                | beam.ParDo(self._MapAndFilterErrorsDoFn(self._fn)))
