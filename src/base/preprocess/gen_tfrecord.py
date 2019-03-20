import logging
import tempfile
import sys

import tensorflow_transform as tft
import apache_beam as beam
from apache_beam.runners import DirectRunner

from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform import coders as tft_coders

from apache_beam.io import textio
from apache_beam.io import tfrecordio

from src.base.preprocess.preprocess_util import MapAndFilterErrors
from src.utils.utils import import_from_uri

log = logging.getLogger('tensorflow')


def write_to_tfrecord(args):
    """
    This function is supposed to be called as a script.
    """
    # Decode arguments
    current_index, num_shards, train_split_fname_out, eval_split_fname_out, \
    train_tfrecord_fname_out, eval_tfrecord_fname_out, working_dir, data_formatter_module_path = args

    # num_shards = "32"
    current_index, num_shards = int(current_index), int(num_shards)

    split_train_file_pattern = '{}-{:05}-of-{:05}'.format(train_split_fname_out, current_index, num_shards) + '*'
    split_eval_file_pattern = '{}-{:05}-of-{:05}'.format(eval_split_fname_out, current_index, num_shards)

    log.info('train_tfrecord_fname_out {}'.format(train_tfrecord_fname_out))
    log.info('eval_tfrecord_fname_out {}'.format(eval_tfrecord_fname_out))
    log.info('split_train_file_pattern {}'.format(split_train_file_pattern))
    log.info('split_eval_file_pattern {}'.format(split_eval_file_pattern))


    data_formatter = import_from_uri(data_formatter_module_path).DataFormatter()

    # Set up the preprocessing pipeline.
    pipeline = beam.Pipeline(runner=DirectRunner())

    with beam_impl.Context(temp_dir=tempfile.mkdtemp()):
        # Read raw data files: CSV format ordered according to the `data_formatter`, that are then converted
        # into a cleaned up format.
        raw_train_data = (
                pipeline
                | 'ReadTrainDataFile' >> textio.ReadFromText(
            split_train_file_pattern,
            skip_header_lines=0)
                | 'DecodeTrainDataCSV' >> MapAndFilterErrors(
            tft_coders.CsvCoder(data_formatter.get_features_and_targets(),
                                data_formatter.get_features_metadata().schema).decode)
        )

        raw_eval_data = (
                pipeline
                | 'ReadEvalDataFile' >> textio.ReadFromText(
            split_eval_file_pattern,
            skip_header_lines=0)
                | 'DecodeEvalDataCSV' >> MapAndFilterErrors(
            tft_coders.CsvCoder(data_formatter.get_features_and_targets(),
                                data_formatter.get_features_metadata().schema).decode)
        )

        # Examples in tf-example format (for model analysis purposes).
        # raw_feature_spec = data_formatter.RAW_DATA_METADATA.schema.as_feature_spec()
        # raw_schema = dataset_schema.from_feature_spec(raw_feature_spec)
        # coder = example_proto_coder.ExampleProtoCoder(raw_schema)
        #
        # _ = (
        #         raw_eval_data
        #         | 'ToSerializedTFExample' >> beam.Map(coder.encode)
        #         | 'WriteAnalysisTFRecord' >> tfrecordio.WriteToTFRecord(
        #     '{}-{:05}-of-{:05}'.format(analysis_fname_out, i, num_shards),
        #     shard_name_template='', num_shards=1)
        # )

        # Write SavedModel and metadata to two subdirectories of working_dir, given by
        # `transform_fn_io.TRANSFORM_FN_DIR` and `transform_fn_io.TRANSFORMED_METADATA_DIR` respectively.
        transform_fn = (
                pipeline
                | 'ReadTransformGraph' >>
                transform_fn_io.ReadTransformFn(working_dir))

        # Applies the transformation `transform_fn` to the raw eval dataset
        (transformed_train_data, transformed_metadata) = (
                ((raw_train_data, data_formatter.get_features_metadata()), transform_fn)
                | 'TransformTrainData' >> beam_impl.TransformDataset()
        )

        # Applies the transformation `transform_fn` to the raw eval dataset
        (transformed_eval_data, transformed_metadata) = (
                ((raw_eval_data, data_formatter.get_features_metadata()), transform_fn)
                | 'TransformEvalData' >> beam_impl.TransformDataset()
        )

        # The data schema of the transformed data gets used to build a signature to create
        # a TFRecord (tf binary data format). This signature is a wrapper function used to
        # encode transformed data.
        transformed_data_coder = tft.coders.ExampleProtoCoder(transformed_metadata.schema)

        _ = (
                transformed_train_data
                | 'EncodeTrainDataTransform' >> MapAndFilterErrors(transformed_data_coder.encode)
                | 'WriteTrainDataTFRecord' >> tfrecordio.WriteToTFRecord(
            '{}-{:05}-of-{:05}'.format(train_tfrecord_fname_out, current_index, num_shards),
            shard_name_template='', num_shards=1)
        )

        _ = (
                transformed_eval_data
                | 'EncodeEvalDataTransform' >> MapAndFilterErrors(transformed_data_coder.encode)
                | 'WriteEvalDataTFRecord' >> tfrecordio.WriteToTFRecord(
            '{}-{:05}-of-{:05}'.format(eval_tfrecord_fname_out, current_index, num_shards),
            shard_name_template='', num_shards=1)
        )

    result = pipeline.run()
    result.wait_until_finish()

    # After transforming, remove original files.
    # for fl in glob.glob(split_train_file_pattern):
    #     log.info('Removing {}'.format(fl))
    #     os.remove(fl)


if __name__ == '__main__':
    write_to_tfrecord(sys.argv[1:])
