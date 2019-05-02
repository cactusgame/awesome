import json
import sys
import sqlite3 as lite
from src.extract.feature_definition import feature_extractor_definition
from src.extract.feature_definition import TYPE_INFER


class FeatureSDKCsvImpl():
    def __init__(self, recreate_table=True):
        self.recreate_table = recreate_table
        self.table_name = "FEATURE"
        self.connection = self.__get_connection()  # using sqllite, the storage is a db connection

        if recreate_table:
            self.__update_feature_table_columns()

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

    def save2(self, dt, share_id, feature_names, values):
        '''
        you must call `commit` after 'save'
        the len(feature_names) must equals to len(values)
        '''
        assert len(feature_names) == len(values)
        # print("save ", dt, share_id, feature_names, values)

        c = self.connection.cursor()
        sql_values = ["'{}'".format(x) if type(x) is str else str(x) for x in values]  # convert the str to 'str'
        sql = "INSERT INTO {} (time,share_id,{}) VALUES ({},{},{})".format(
            self.table_name,
            ",".join(feature_names),
            "'" + dt + "'",
            "'" + share_id + "'",
            ",".join(sql_values))
        print(">>>> " + sql)
        c.execute(sql)

    def commit(self):
        self.connection.commit()

    def get(self):
        c = self.connection.cursor()
        cursor = c.execute("SELECT * FROM {}".format(self.table_name))
        values = cursor.fetchall()
        print(values)

    # ----- private -----
    def __get_connection(self):
        try:
            con = lite.connect('awesome.db')
        except lite.Error, e:
            print "Error %s:" % e.args[0]
            sys.exit(1)
        return con

    def __update_feature_table_columns(self):
        """
        update the feature table in feature.db
        according the `feature_definition` file
        :return:
        """
        self.__init_table()

        # filter out field need to store into db
        feature_dict_sorted_keys = [key for key in feature_extractor_definition.keys() if
                                    feature_extractor_definition[key][5] != TYPE_INFER]
        feature_dict_sorted_keys.sort()
        for key in feature_dict_sorted_keys:
            if not self.__has_feature_column(key):
                self.__add_feature_column(key, feature_extractor_definition[key])

    def __init_table(self):
        try:
            c = self.connection.cursor()
            if self.recreate_table:
                c.execute("DROP TABLE IF EXISTS {};".format(self.table_name))
            c.execute('''CREATE TABLE IF NOT EXISTS {}
                   (time        CHAR(25)    NOT NULL,
                   share_id     CHAR(15)    NOT NULL,
                   primary key(TIME,SHARE_ID))  ;'''.format(self.table_name))
        except Exception, err:
            print(err)

    def __has_feature_column(self, key):
        c = self.connection.cursor()
        columns = [i[1] for i in c.execute('PRAGMA table_info({})'.format(self.table_name))]
        if key in columns:
            return True
        return False

    def __add_feature_column(self, key, value_type):
        print("to add column name={} type={}".format(key, value_type[1]))
        c = self.connection.cursor()
        c.execute('ALTER TABLE {} ADD COLUMN {} {}'.format(self.table_name, key, value_type[1]))


if __name__ == "__main__":
    # test case
    sdk = FeatureSDK(recreate_table=False)
    sdk.get()
