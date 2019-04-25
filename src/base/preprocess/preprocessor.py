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

from src.config.app_config import app_config

from src.utils.file_util import FileUtil
from src.base.config import cfg
from src.base.preprocess.preprocess_util import MapAndFilterErrors
from src.base.preprocess.preprocess_util import PreprocessingFunction
from src.utils.utils import import_from_uri
from src.context import log

"""
to preprocess the experiment log

get the .csv file's header(column name) first, keep it in somewhere
"""


class Preprocessor:
    def __init__(self):
        self.exp_log_data_file_shuf = None
        self.exp_log_header = None
        self.data_formatter = import_from_uri(cfg.cls_data_formatter).DataFormatter()

    def process(self):
        self.reset_env()

        self.download_features_db()

        self.select_features()

        self.shuf()

        self.divide_train_eval()

        self.split_to_shards()

        self.build_graph()

        self.transform()

    def reset_env(self):
        if os.path.exists(cfg.TARGET_DIR):
            shutil.rmtree(cfg.TARGET_DIR)
        os.makedirs(cfg.TARGET_DIR)

    def download_features_db(self):
        """
        1. donwload the awesome.db from COS
        2. export to .csv
        :return:
        """
        FileUtil.download_data("/dv/data/awesome.db", "awesome.db")

        conn = sqlite3.connect('awesome.db')
        cursor = conn.cursor()
        cursor.execute("select * from FEATURE;")

        csv_file = FileUtil.open_file("data/features_db.csv", "wb")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([i[0] for i in cursor.description])  # write headers
        csv_writer.writerows(cursor)
        csv_file.close()

    def select_features(self):
        """
        according to `DataFormatter`, select features used in the algorithm and add target
        :return:
        """
        coder = import_from_uri(cfg.cls_coder).Coder(self.data_formatter)

        output = open(cfg.exp_file_path, 'w')
        writer = csv.DictWriter(output, fieldnames=self.data_formatter.get_features_and_targets(), delimiter=',')

        input_file = open(cfg.feature_db_path, 'rb')
        reader = csv.DictReader(input_file, delimiter=',')

        writer.writeheader()
        for row in reader:
            line = coder.parse(row)
            writer.writerow(line)

        input_file.close()
        output.close()

    def shuf(self):
        st = time.time()
        log.info('shuff start')

        # Use terashuf quasi-shuffle
        FileUtil.save_remove_first_line(cfg.exp_file_path,cfg.exp_log_data_file_shuf_tmp)

        shuf_cmd = 'MEMORY={:.1f} terashuf < {} > {}'.format(cfg.SHUF_MEM, cfg.exp_log_data_file_shuf_tmp,
                                                             cfg.exp_log_data_file_shuf)
        log.info('Executing shuf call: \"{}\"'.format(shuf_cmd))

        ret_stat = os.system(shuf_cmd)
        if ret_stat != 0:
            log.info('`terashuf` failed, falling back to `sort -R`.')
            shuf_cmd = 'sort -R {} -o {}'.format(cfg.exp_file_path, cfg.exp_log_data_file_shuf)
            ret_stat = os.system(shuf_cmd)

        log.info('Executing shuf call: \"{}\"'.format(shuf_cmd))
        log.info("complete shuff. use time {:.2f}s".format(time.time() - st))

    def divide_train_eval(self):
        st = time.time()
        log.info(
            'Splitting data file into train ({}) and eval ({}) set.'.format(cfg.exp_log_data_file_train,
                                                                            cfg.exp_log_data_file_eval))
        assert cfg.TRAIN_SPLIT_RATIO < 1.0
        assert cfg.TRAIN_SPLIT_RATIO > 0.5

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
        splitter = mp.Process(
            target=divide_file,
            args=(cfg.exp_log_data_file_shuf,
                  cfg.exp_log_data_file_train,
                  cfg.exp_log_data_file_eval,
                  cfg.TRAIN_SPLIT_RATIO)
        )
        splitter.start()
        splitter.join()

        # os.remove(self.rec_log_data_file_shuf)
        log.info('complete data split in {:.2f} sec.'.format(time.time() - st))

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
        assert isinstance(cfg.DATASET_NUM_SHARDS, int)

        train_splitter = mp.Process(
            target=split_file,
            args=(cfg.exp_log_data_file_train, cfg.exp_log_data_file_train_shard, cfg.DATASET_NUM_SHARDS))
        eval_splitter = mp.Process(
            target=split_file,
            args=(cfg.exp_log_data_file_eval, cfg.exp_log_data_file_eval_shard, cfg.DATASET_NUM_SHARDS))
        train_splitter.start()
        eval_splitter.start()
        train_splitter.join()
        eval_splitter.join()

        # After splitting, remove original shuffed file.
        # os.remove(self.train_split_fname_out)
        # os.remove(self.eval_split_fname_out)
        log.info(
            'complete split Train and Eval files into serval shards. use {:.2f} sec.'.format(time.time() - st))

    def build_graph(self):
        # Set up the preprocessing pipeline for analyzing the dataset. The analyze call is not combined with the
        # transform call because we will parallelize the transform call later. We had the issue that this process
        # runs on a single core and tends to cause OOM issues.
        pipeline = beam.Pipeline(runner=DirectRunner())

        with beam_impl.Context(temp_dir=tempfile.mkdtemp()):
            # todo: I MUST only use train data (or percentage of train data) to build the graph
            # if you add the eval set to generate the GRAPH, it will affect eval result
            raw_train_data = (
                    pipeline
                    | 'ReadTrainDataFile' >> textio.ReadFromText('data/features' + '*' + 'shard' + '*',
                                                                 skip_header_lines=0)
                    | 'DecodeTrainDataCSV' >> MapAndFilterErrors(
                tft_coders.CsvCoder(self.data_formatter.get_features_and_targets(),
                                    self.data_formatter.features_metadata.schema).decode)
            )

            # Combine data and schema into a dataset tuple.  Note that we already used
            # the schema to read the CSV data, but we also need it to interpret
            # raw_data.
            # That is when to use vocabulary, scale_to_0_1 or sparse_to_dense ...
            transform_fn = (
                    (raw_train_data, self.data_formatter.features_metadata)
                    | beam_impl.AnalyzeDataset(PreprocessingFunction(self.data_formatter).transform_to_tfrecord))

            # Write SavedModel and metadata to two subdirectories of working_dir, given by
            # `transform_fn_io.TRANSFORM_FN_DIR` and `transform_fn_io.TRANSFORMED_METADATA_DIR` respectively.
            _ = (
                    transform_fn
                    | 'WriteTransformGraph' >>
                    transform_fn_io.WriteTransformFn(cfg.TARGET_DIR))  # working dir

        # Run the Beam preprocessing pipeline.
        st = time.time()
        result = pipeline.run()
        result.wait_until_finish()
        log.info('Transformation graph built and written in {:.2f} sec'.format(time.time() - st))

    def transform(self):
        """
        transform to tfrecord
        :return:
        """
        import subprocess
        st = time.time()
        _exec_path = os.path.abspath(os.path.join("src/base/preprocess", "gen_tfrecord.py"))

        results = list()
        num_proc = mp.cpu_count() / 2
        for i in range(cfg.DATASET_NUM_SHARDS):
            log.info('Running transformer pipeline {}.'.format(i))

            data_formatter_module_path = cfg.cls_data_formatter

            python_command = app_config.SUBPROCESS_PYTHON  # Notice: the python must the same python as the master process
            call = [python_command, _exec_path, str(i), str(cfg.DATASET_NUM_SHARDS),
                    cfg.exp_log_data_file_train_shard, cfg.exp_log_data_file_eval_shard,
                    cfg.exp_log_data_file_train_tfrecord, cfg.exp_log_data_file_eval_tfrecord, cfg.TARGET_DIR,data_formatter_module_path]
            log.info("Sub process command to transform: {}".format(call))

            results.append(subprocess.Popen(call))

            # I don't know why it must sleep a while. otherwise, the subprocess will be skipped.
            time.sleep(5)

            if (i + 1) % num_proc == 0:
                # block exec after
                for result in results:
                    while result.poll() is None:
                        pass
                    if result.returncode != 0:
                        log.error('Transformer pipeline return code: {}. '
                                  'Hint: when running on GPU, set `num_proc=1`.'.format(result.returncode))
                        raise Exception('Transformer pipeline return code: {}. '
                                        'Hint: when running on GPU, set `num_proc=1`.'.format(result.returncode))

        log.info('Finished transforming train/eval sets to TFRecord in {:.2f} sec.'.format(time.time() - st))
