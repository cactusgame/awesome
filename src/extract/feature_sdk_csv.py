import csv
import os
from src.extract.feature_definition import feature_extractor_definition
from src.extract.feature_definition import TYPE_INFER
from src.utils.file_util import FileUtil
from src.extract.feature_definition import DOWNLOAD_FEATURES
from src.extract.econfig import econfig


class FeatureSDKCsvImpl:
    def __init__(self, feature_data_source):
        # create csv file
        self.feature_columns = None
        self.csv_writer = None
        self.feature_data_source = feature_data_source
        self.feature_data_csv = "{}.csv".format(feature_data_source)
        self.csvfile = open(self.feature_data_csv, 'w')
        self.enable_upload = False if econfig.DEBUG else True

    def init_storage(self):
        self._update_feature_table_columns()
        # create the csv file writer
        self.csv_writer = csv.DictWriter(self.csvfile, fieldnames=self.feature_columns)
        self.csv_writer.writeheader()

    def save(self, feature_names, values):
        assert len(feature_names) == len(values)

        line = {}
        for i in range(len(feature_names)):
            v = values[i]
            # adjust the precision for float
            if feature_extractor_definition[feature_names[i]][0] == 'float':
                v = round(v, 4)
            line[feature_names[i]] = v

        self.csv_writer.writerow(line)

    def commit(self):
        """
        1 share extraction complete
        :return:
        """
        pass

    def close(self):
        """
        all shares extraction complete
        :return:
        """
        if self.enable_upload or True:
            FileUtil.upload_data(os.path.abspath(self.feature_data_csv))
        self.csvfile.close()

    def get(self):
        with open(self.feature_data_csv, 'r') as csvfile:
            for line in csvfile:
                print(line)

    @staticmethod
    def download():
        # download all parts of features
        for feature_file_name in DOWNLOAD_FEATURES:
            FileUtil.download_data("data/{}.csv".format(feature_file_name), "{}.csv".format(feature_file_name))

    def _update_feature_table_columns(self):
        """
        update the feature table in feature.db
        according the `feature_definition` file
        :return:
        """
        # filter out field need to store into db
        feature_columns = [key for key in feature_extractor_definition.keys() if
                           feature_extractor_definition[key][5] != TYPE_INFER]
        feature_columns.sort()
        self.feature_columns = feature_columns


if __name__ == "__main__":
    sdk = FeatureSDKCsvImpl()
    sdk.get()
