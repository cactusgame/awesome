import time
import os

from src.context import context
from src.utils.file_util import FileUtil


class Preprocessor:
    def __init__(self, feature_file_path):
        self.logger = context.logger
        self.exp_file_path = exp_file_path

    def process(self):
        self.shuf()

        self.divide_train_eval()

        self.split_to_block()

        self.build_graph()

        self.transform()

    def shuf(self):
        st = time.time()
        self.logger.info('shuff start')

        SHUF_MEM = 1  # the shuf command memory usage Gi

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

    def divide_train_eval(self):
        pass

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
