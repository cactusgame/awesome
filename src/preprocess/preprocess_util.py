import apache_beam as beam
import tensorflow_transform as tft

from src.context import context
from src.extract.feature_definition import *

class PreprocessingFunction(object):
    """
    A transformer that preprocessed data in csv into TFT format columns
    """

    def __init__(self):
        pass

    def transform_to_tf(self, inputs):
        """Preprocess raw input columns into transformed columns."""
        outputs = inputs.copy()

        for key in number_keys:
            outputs[key] = tft.scale_to_z_score((outputs[key]))

        # for key in OPTIONAL_NUMERIC_FEATURE_KEYS:
        #     # This is a SparseTensor because it is optional. Here we fill in a default
        #     # value when it is missing.
        #     dense = tf.sparse_to_dense(outputs[key].indices,
        #                                [outputs[key].dense_shape[0], 1],
        #                                outputs[key].values, default_value=0.)
        #     # Reshaping from a batch of vectors of size 1 to a batch to scalars.
        #     dense = tf.squeeze(dense, axis=1)
        #     outputs[key] = tft.scale_to_0_1(dense)

        for key in vocabulary_keys:
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
