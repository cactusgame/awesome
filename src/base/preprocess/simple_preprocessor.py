import time
import os
import multiprocessing as mp
import shutil

from src.utils.file_util import FileUtil
from src.base.config import cfg
from src.context import log

"""
to preprocess the experiment log
this preprocess is responsible for
- download feature file
- by train and eval
    - shuffle
    - split to shard
    - transform to TFR

get the .csv file's header(column name) first, keep it in somewhere
"""


class SimplePreprocessor:
    def __init__(self):
        self.exp_log_data_file_shuf = None
        self.exp_log_header = None
        # self.data_formatter = import_from_uri(cfg.cls_data_formatter).DataFormatter()
        self.files = ['feature_train', 'feature_eval']

    def process(self):
        if not cfg.do_preprocessing:
            return

        self.reset_env()

        print("start to download feature db")
        if cfg.download_feature_db:
            self.download_features_db()

        for file in self.files:
            self.shuf(file)

            self.split_to_shards(file)

            # self.transform(file)

    def reset_env(self):
        if os.path.exists(cfg.TARGET_DIR):
            shutil.rmtree(cfg.TARGET_DIR)
        os.makedirs(cfg.TARGET_DIR)

    def download_features_db(self):
        """
        the download process is implemented in the feature SDK
        1. donwload the xxx.csv from COS
        :return:
        """
        for feature_file_name in self.files:
            FileUtil.download_data("data/{}.csv".format(feature_file_name), "data/{}.csv".format(feature_file_name))

    def shuf(self, file_name):
        st = time.time()
        log.info('shuff start')

        # Use terashuf quasi-shuffle
        FileUtil.save_remove_first_line(cfg.get_exp_file(file_name), cfg.get_exp_file_without_header(file_name))

        shuf_cmd = 'MEMORY={:.1f} terashuf < {} > {}'.format(cfg.SHUF_MEM, cfg.get_exp_file_without_header(file_name),
                                                             cfg.get_shuffled_file(file_name))
        log.info('Executing shuf call: \"{}\"'.format(shuf_cmd))

        ret_stat = os.system(shuf_cmd)
        if ret_stat != 0:
            log.info('`terashuf` failed, falling back to `sort -R`.')
            shuf_cmd = 'sort -R {} -o {}'.format(cfg.get_exp_file_without_header(file_name), cfg.get_shuffled_file(file_name))
            ret_stat = os.system(shuf_cmd)

        log.info('Executing shuf call: \"{}\"'.format(shuf_cmd))
        log.info("complete shuff. use time {:.2f}s".format(time.time() - st))

    def split_to_shards(self, file_name):
        # print("start to split into shards")

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

        splitter = mp.Process(
            target=split_file,
            args=(cfg.get_shuffled_file(file_name), cfg.get_shard_file(file_name), cfg.DATASET_NUM_SHARDS))
        splitter.start()
        splitter.join()

        # After splitting, remove original shuffed file.
        # os.remove(self.train_split_fname_out)
        # os.remove(self.eval_split_fname_out)
        log.info(
            'complete split Train and Eval files into serval shards. use {:.2f} sec.'.format(time.time() - st))

