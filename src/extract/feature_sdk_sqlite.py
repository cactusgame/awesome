import json
import sys
import sqlite3 as lite
from src.extract.feature_definition import feature_extractor_definition
from src.extract.feature_definition import TYPE_INFER


class FeatureSDKSqliteImpl():
    def __init__(self):
        self.table_name = "FEATURE"
        self.connection = self._get_connection()  # using sqllite, the storage is a db connection

    def init_storage(self):
        self._update_feature_table_columns()

    def save(self, feature_names, values):
        assert len(feature_names) == len(values)

        c = self.connection.cursor()
        sql_values = ["'{}'".format(x) if type(x) is str else str(x) for x in values]  # convert the str to 'str'
        sql = "INSERT INTO {table_name} ({feature_names}) VALUES ({values})".format(
            table_name=self.table_name,
            feature_names=",".join(feature_names),
            values=",".join(sql_values))
        # print(">>>> " + sql)
        c.execute(sql)

    def commit(self):
        self.connection.commit()

    def close(self):
        pass

    def get_all(self):
        c = self.connection.cursor()
        cursor = c.execute("SELECT * FROM {} limit 10".format(self.table_name))
        values = cursor.fetchall()
        print(values)

    def _get_connection(self):
        try:
            con = lite.connect('awesome.db')
        except lite.Error, e:
            print "Error %s:" % e.args[0]
            sys.exit(1)
        return con

    def _update_feature_table_columns(self):
        """
        update the feature table in feature.db
        according the `feature_definition` file
        :return:
        """
        self._init_table()

        # filter out field need to store into db
        feature_dict_sorted_keys = [key for key in feature_extractor_definition.keys() if
                                    feature_extractor_definition[key][5] != TYPE_INFER]
        feature_dict_sorted_keys.sort()
        for key in feature_dict_sorted_keys:
            if not self._has_feature_column(key):
                self._add_feature_column(key, feature_extractor_definition[key])

    def _init_table(self):
        try:
            c = self.connection.cursor()
            c.execute("DROP TABLE IF EXISTS {};".format(self.table_name))
            c.execute('''CREATE TABLE IF NOT EXISTS {}
                   (time        CHAR(25)    NOT NULL,
                   share_id     CHAR(15)    NOT NULL,
                   primary key(TIME,SHARE_ID))  ;'''.format(self.table_name))
        except Exception, err:
            print(err)

    def _has_feature_column(self, key):
        c = self.connection.cursor()
        columns = [i[1] for i in c.execute('PRAGMA table_info({})'.format(self.table_name))]
        if key in columns:
            return True
        return False

    def _add_feature_column(self, key, value_type):
        print("to add column name={} type={}".format(key, value_type[1]))
        c = self.connection.cursor()
        c.execute('ALTER TABLE {} ADD COLUMN {} {}'.format(self.table_name, key, value_type[1]))


if __name__ == "__main__":
    # test case
    sdk = FeatureSDKSqliteImpl()
    sdk.get_all()
