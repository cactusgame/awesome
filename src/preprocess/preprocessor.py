import time
import os
import multiprocessing as mp
import numpy as np
import apache_beam as beam
import tempfile
import shutil
import csv
import sqlite3

from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform import coders as tft_coders
from apache_beam.io import textio
from apache_beam.runners import DirectRunner
from src.context import context
from src.extract.feature_definition import new_feature_column_names
from src.extract.feature_definition import new_feature_column_functions
from src.utils.file_util import FileUtil
from src.training.train_config import *
from src.preprocess.preprocess_util import MapAndFilterErrors
from src.preprocess.preprocess_util import PreprocessingFunction
from src.preprocess.data_formatter import DataFormatter
from src.config.app_config import app_config

"""
to preprocess the experiment log

get the .csv file's header(column name) first, keep it in somewhere
"""


class Preprocessor:
    def __init__(self):
        self.logger = context.logger
        self.exp_log_data_file_shuf = None
        self.exp_log_data_file_without_header = None
        self.exp_log_header = None

        self.data_formatter = None

    def process(self):
        self.reset_env()

        self.download_features()

        self.add_target_keys()

        self.shuf()

        self.divide_train_eval()

        self.split_to_shards()

        self.build_graph()

        self.transform()

    def reset_env(self):
        if os.path.exists(TARGET_DIR):
            shutil.rmtree(TARGET_DIR)
        os.makedirs(TARGET_DIR)

    def download_features(self):
        """
        1. donwload the awesome.db from COS
        2. export to .csv
        :return:
        """
        FileUtil.download_data("/dv/data/awesome.db", "awesome.db")

        conn = sqlite3.connect('awesome.db')
        cursor = conn.cursor()
        cursor.execute("select * from FEATURE;")
        # with open("out.csv", "w", newline='') as csv_file:  # Python 3 version
        with open("data/features.csv", "wb") as csv_file:  # Python 2 version
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([i[0] for i in cursor.description])  # write headers
            csv_writer.writerows(cursor)

    def add_target_keys(self):
        """
        in order to generate the TARGET key in training. add target keys firstly
        :return:
        """
        # output
        output = open(exp_target_file_path, 'w')
        writer = csv.writer(output, delimiter=',')

        inputfile = open(exp_file_path, 'rb')
        reader = csv.reader(inputfile, delimiter=',')

        header = None
        for row in reader:
            if header is None:
                for feature_column_name in new_feature_column_names:
                    row.append(feature_column_name)
                header = row
            else:
                for feature_column_function in new_feature_column_functions:
                    row.append(feature_column_function(header, row))
            writer.writerow(row)
        inputfile.close()
        output.close()

    def shuf(self):
        st = time.time()
        self.logger.info('shuff start')

        # we need save the .csv file header first and insert it after shuf
        exp_log_header = FileUtil.save_remove_first_line(exp_log_data_file, exp_log_data_file_without_header)

        self.data_formatter = DataFormatter()
        self.data_formatter.init_columns(exp_log_header)

        # Use terashuf quasi-shuffle
        shuf_cmd = 'MEMORY={:.1f} terashuf < {} > {}'.format(SHUF_MEM, exp_log_data_file_without_header,
                                                             exp_log_data_file_shuf)
        self.logger.info('Executing shuf call: \"{}\"'.format(shuf_cmd))

        ret_stat = os.system(shuf_cmd)
        if ret_stat != 0:
            self.logger.info('`terashuf` failed, falling back to `sort -R`.')
            shuf_cmd = 'sort -R {} -o {}'.format(exp_log_data_file_without_header, exp_log_data_file_shuf)
            ret_stat = os.system(shuf_cmd)

        # os.remove(exp_log_data_file_without_header)
        self.logger.info('Executing shuf call: \"{}\"'.format(shuf_cmd))
        self.logger.info("complete shuff. use time {:.2f}s".format(time.time() - st))

        self.exp_log_data_file_shuf = exp_log_data_file_shuf
        self.exp_log_data_file_without_header = exp_log_data_file_without_header
        self.exp_log_header = exp_log_header

    def divide_train_eval(self):
        st = time.time()
        self.logger.info(
            'Splitting data file into train ({}) and eval ({}) set.'.format(exp_log_data_file_train,
                                                                            exp_log_data_file_eval))
        assert TRAIN_SPLIT_RATIO < 1.0
        assert TRAIN_SPLIT_RATIO > 0.5

        def divide_file(fname_in, out_file_a_name, out_file_b_name, a_split_ratio):
            f_in = open(fname_in, 'rb')

            out_file_a = open(os.path.abspath(out_file_a_name), 'w+')
            out_file_b = open(os.path.abspath(out_file_b_name), 'w+')

            for line in f_in:
                if np.random.rand() < a_split_ratio:
                    out_file_a.write(line)
                else:
                    out_file_b.write(line)

            out_file_a.close()
            out_file_b.close()
            f_in.close()

        # To avoid locking memory
        train_splitter = mp.Process(
            target=divide_file,
            args=(self.exp_log_data_file_without_header,
                  exp_log_data_file_train,
                  exp_log_data_file_eval,
                  TRAIN_SPLIT_RATIO)
        )
        train_splitter.start()
        train_splitter.join()

        # os.remove(self.rec_log_data_file_shuf)
        self.logger.info('complete data split in {:.2f} sec.'.format(time.time() - st))

    def split_to_shards(self):
        def split_file(fname_in, fname_out, num_shards):
            f_outs = list()
            for i in range(num_shards):
                f_outs.append(open('{}-{:05}-of-{:05}'.format(fname_out, i, num_shards), 'w+'))
            f_in = open(fname_in, 'rb')
            r_c = 0
            for line in f_in:
                f_outs[r_c].write(line)
                r_c = (r_c + 1) % num_shards
            for f_out in f_outs:
                f_out.close()
            f_in.close()

        st = time.time()
        assert isinstance(DATASET_NUM_SHARDS, int)

        train_splitter = mp.Process(
            target=split_file,
            args=(exp_log_data_file_train, exp_log_data_file_train_shard, DATASET_NUM_SHARDS))
        eval_splitter = mp.Process(
            target=split_file,
            args=(exp_log_data_file_eval, exp_log_data_file_eval_shard, DATASET_NUM_SHARDS))
        train_splitter.start()
        eval_splitter.start()
        train_splitter.join()
        eval_splitter.join()

        # After splitting, remove original shuffed file.
        # os.remove(self.train_split_fname_out)
        # os.remove(self.eval_split_fname_out)
        self.logger.info(
            'complete split Train and Eval files into serval shards. use {:.2f} sec.'.format(time.time() - st))

    def build_graph(self):
        # Move percentage of train data to .`PPGRAPH_EXT` files, used for graph building.
        # num_lines = 0
        # for i in range(DATASET_NUM_SHARDS):
        #     _fname = '{}-{:05}-of-{:05}'.format(self.train_fname_out, i, self.config.DATASET_NUM_SHARDS)
        #     num_lines += sum(1 for _ in open(_fname))
        #     _fname_marked = '{}-{:05}-of-{:05}.{}'.format(self.train_fname_out, i, self.config.DATASET_NUM_SHARDS,
        #                                                   PPGRAPH_EXT)
        #     shutil.move(_fname, _fname_marked)
        #     if num_lines >= self.config.PPGRAPH_MAX_SAMPLES:
        #         break

        # Set up the preprocessing pipeline for analyzing the dataset. The analyze call is not combined with the
        # transform call because we will parallelize the transform call later. We had the issue that this process
        # runs on a single core and tends to cause OOM issues.
        pipeline = beam.Pipeline(runner=DirectRunner())

        with beam_impl.Context(temp_dir=tempfile.mkdtemp()):
            # todo: maybe, I should only use train data (or percentage of train data) to build the graph
            raw_train_data = (
                    pipeline
                    | 'ReadTrainDataFile' >> textio.ReadFromText('data/features' + '*' + 'shard' + '*',
                                                                 skip_header_lines=0)
                    | 'DecodeTrainDataCSV' >> MapAndFilterErrors(
                tft_coders.CsvCoder(self.data_formatter.get_ordered_columns(),
                                    self.data_formatter.get_raw_data_metadata().schema).decode)
            )

            # Combine data and schema into a dataset tuple.  Note that we already used
            # the schema to read the CSV data, but we also need it to interpret
            # raw_data.
            # That is when to use vocabulary, scale_to_0_1 or sparse_to_dense ...
            transform_fn = (
                    (raw_train_data, self.data_formatter.get_raw_data_metadata())
                    | beam_impl.AnalyzeDataset(PreprocessingFunction().transform_to_tfrecord))

            # Write SavedModel and metadata to two subdirectories of working_dir, given by
            # `transform_fn_io.TRANSFORM_FN_DIR` and `transform_fn_io.TRANSFORMED_METADATA_DIR` respectively.
            _ = (
                    transform_fn
                    | 'WriteTransformGraph' >>
                    transform_fn_io.WriteTransformFn(TARGET_DIR))  # working dir

        # Run the Beam preprocessing pipeline.
        st = time.time()
        result = pipeline.run()
        result.wait_until_finish()
        self.logger.info('Transformation graph built and written in {:.2f} sec'.format(time.time() - st))

    def transform(self):
        """
        transform to tfrecord
        :return:
        """
        import subprocess
        st = time.time()
        _exec_path = os.path.abspath(os.path.join("src/preprocess", "gen_tfrecord.py"))

        results = list()
        num_proc = mp.cpu_count() / 2
        for i in range(DATASET_NUM_SHARDS):
            self.logger.info('Running transformer pipeline {}.'.format(i))

            python_command = app_config.SUBPROCESS_PYTHON  # Notice: the python must the same python as the master process
            call = [python_command, _exec_path, self.exp_log_header, str(i), str(DATASET_NUM_SHARDS),
                    exp_log_data_file_train_shard, exp_log_data_file_eval_shard,
                    train_tfrecord_fname_out, eval_tfrecord_fname_out, TARGET_DIR]
            self.logger.info("Sub process command to transform: {}".format(call))

            results.append(subprocess.Popen(call))

            # I don't know why it must sleep a while. otherwise, the subprocess will be skipped.
            time.sleep(5)

            if (i + 1) % num_proc == 0:
                # block exec after
                for result in results:
                    while result.poll() is None:
                        pass
                    if result.returncode != 0:
                        self.logger.error('Transformer pipeline return code: {}. '
                                          'Hint: when running on GPU, set `num_proc=1`.'.format(result.returncode))
                        raise Exception('Transformer pipeline return code: {}. '
                                        'Hint: when running on GPU, set `num_proc=1`.'.format(result.returncode))

        self.logger.info('Finished transforming train/eval sets to TFRecord in {:.2f} sec.'.format(time.time() - st))
        return train_tfrecord_fname_out, eval_tfrecord_fname_out


if __name__ == "__main__":
    _start = time.time()

    preprocessor = Preprocessor()
    preprocessor.process()

    context.logger.info("[total] preprocess use time {:.2f}s".format(time.time() - _start))
