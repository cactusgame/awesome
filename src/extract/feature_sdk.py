from src.extract.feature_sdk_sqlite import FeatureSDKSqliteImpl
from src.extract.feature_sdk_csv import FeatureSDKCsvImpl


class FeatureSDK:

    @staticmethod
    def download():
        FeatureSDKCsvImpl.download()
        # FeatureSDKSqliteImpl.download()

    def __init__(self, feature_data_source):
        # self.impl = FeatureSDKSqliteImpl()
        self.impl = FeatureSDKCsvImpl(feature_data_source)

        self.init_storage()

    def init_storage(self):
        """
        currently, the method will re-create the file, table or something.
        in the future, you may only append new data but without recreate the storage
        :return:
        """
        self.impl.init_storage()

    def save(self, feature_names, values):
        """
        save the specific values with the specific column names
        the save process is only save into memory or buffer, you must call `commit` to flush the saved content
        :param feature_names: a list of feature names
        :param values: a list of values
        :return:
        """
        self.impl.save(feature_names, values)

    def commit(self):
        """
        flush the saved content, upload
        :return:
        """
        self.impl.commit()

    def close(self):
        """
        when close the sdk
        :return:
        """
        self.impl.close()
