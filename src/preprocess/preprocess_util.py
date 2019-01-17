import apache_beam as beam

from src.context import context


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
