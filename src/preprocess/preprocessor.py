import time
import os
import multiprocessing as mp
import numpy as np

from src.context import context
from src.utils.file_util import FileUtil
from src.preprocess.preprocess_config import *


class Preprocessor:
    def __init__(self, feature_file_path):
        self.logger = context.logger
        self.exp_file_path = exp_file_path
        self.exp_log_data_file_shuf = None

    def process(self):
        self.shuf()

        self.divide_train_eval()

        self.split_to_block()

        self.build_graph()

        self.transform()

    def shuf(self):
        st = time.time()
        self.logger.info('shuff start')

        # we need save the .csv file header first and insert it after shuf
        exp_log_data_file = os.path.abspath(self.exp_file_path)
        exp_log_data_file_without_header = '{}.withoutheader'.format(exp_log_data_file)
        exp_log_data_file_shuf = '{}.shuf'.format(exp_log_data_file)

        exp_log_header = FileUtil.save_remove_first_line(exp_log_data_file, exp_log_data_file_without_header)

        # Use terashuf quasi-shuffle
        shuf_cmd = 'MEMORY={:.1f} terashuf < {} > {}'.format(SHUF_MEM, exp_log_data_file_without_header,
                                                             exp_log_data_file_shuf)
        self.logger.info('Executing shuf call: \"{}\"'.format(shuf_cmd))

        ret_stat = os.system(shuf_cmd)
        if ret_stat != 0:
            self.logger.info('`terashuf` failed, falling back to `sort -R`.')
            shuf_cmd = 'sort -R {} -o {}'.format(exp_log_data_file_without_header, exp_log_data_file_shuf)
            ret_stat = os.system(shuf_cmd)

        FileUtil.add_header_to_file(exp_log_header, exp_log_data_file_shuf)

        os.remove(exp_log_data_file_without_header)
        self.logger.info('Executing shuf call: \"{}\"'.format(shuf_cmd))
        self.logger.info("shuff complete. use time {:.2f}s".format(time.time() - st))

        self.exp_log_data_file_shuf = exp_log_data_file_shuf
        self.exp_log_data_file_without_header = exp_log_data_file_without_header

    def divide_train_eval(self):
        st = time.time()
        exp_log_data_file_train = '{}.train'.format(self.exp_file_path)
        exp_log_data_file_eval = '{}.eval'.format(self.exp_file_path)

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

        exp_log_header = FileUtil.save_remove_first_line(self.exp_log_data_file_shuf,
                                                         self.exp_log_data_file_without_header)
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

        # add header for train and eval file
        FileUtil.add_header_to_file(exp_log_header, exp_log_data_file_train)
        FileUtil.add_header_to_file(exp_log_header, exp_log_data_file_eval)

        # os.remove(self.rec_log_data_file_shuf)
        os.remove(self.exp_log_data_file_without_header)

        self.logger.info('complete data split in {:.2f} sec.'.format(time.time() - st))

        self.exp_log_data_file_train = exp_log_data_file_train
        self.exp_log_data_file_eval = exp_log_data_file_eval

    def split_to_block(self):
        pass

    def build_graph(self):
        pass

    def transform(self):
        pass


if __name__ == "__main__":
    _start = time.time()

    exp_file_path = "data/features.csv"

    preprocessor = Preprocessor(exp_file_path)
    preprocessor.process()

    print("preprocess use time {:.2f}s".format(time.time() - _start))
