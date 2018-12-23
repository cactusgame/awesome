import json
import sys
import sqlite3 as lite
from pprint import pprint
from feature_definition import feature_definition


class FeatureSDK():
    def __init__(self):
        self.connection = None  # using sqllite, the storage is a db connection

    def update_feature_table_columns(self):
        """
        update the feature table in feature.db
        according the `feature_definition` file
        :return:
        """
        feature_dict = self.load_feature_definition()
        self.connection = self.get_connection()

        self.init_table_lazy()

        for key, value in feature_dict.iteritems():
            if not self.has_feature_column(key):
                self.create_feature_column(key, value)

        self.connection.commit()
        self.connection.close()

    # ----- private -----
    def get_connection(self):
        try:
            con = lite.connect('awesome.db')
        except lite.Error, e:
            print "Error %s:" % e.args[0]
            sys.exit(1)
        return con

    def init_table_lazy(self):
        try:
            c = self.connection.cursor()
            c.execute('''CREATE TABLE FEATURE
                   (time        CHAR(25)    NOT NULL,
                   share_id     CHAR(15)    NOT NULL,
                   primary key(TIME,SHARE_ID))  ;''')
        except Exception, err:
            print(err)

    def has_feature_column(self, key):
        c = self.connection.cursor()
        columns = [i[1] for i in c.execute('PRAGMA table_info(FEATURE)')]
        if key in columns:
            return True
        return False

    def create_feature_column(self, key, value_type):
        print("to create column name={} type={}".format( key , value_type[1]))
        c = self.connection.cursor()
        c.execute('ALTER TABLE FEATURE ADD COLUMN {} {}'.format(key, value_type[1]))

    def load_feature_definition(self):
        """
        load feature definition from a file
        :return:
        """
        return feature_definition


if __name__ == "__main__":
    # test case
    sdk = FeatureSDK()
    sdk.update_feature_table_columns()
