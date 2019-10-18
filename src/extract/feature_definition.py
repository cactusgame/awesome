import tensorflow as tf
import collections

TRAIN_FILE_NAME = 'feature_train_big_bank_20days_hloc'
EVAL_FILE_NAME = 'feature_eval_big_bank_20days_hloc'

# define different type of features, usually, it's depends on the data source or data API
FEATURE_ALL = "feature_all"
FEATURE_BASIC = "feature_basic"
FEATURE_FINANCE = "feature_finance"

DOWNLOAD_FEATURES = [FEATURE_BASIC]

feature_definition_config = {}
feature_definition_config["hloc_seq_step"] = 21  # n -1 is the real `steps`, due to the last one is target
feature_definition_config["ror_n_days_after"] = 60

# feature_extractor_definition is only used for feature extractor,
# items in the `feature_extractor_definition` will create a column in database
feature_extractor_definition = collections.OrderedDict()
# format:
# key: feature name
# value:
# [0] type in python
# [1] type in sqlite
# [2] feature def in TFT
# [3] feature preprocessing type in TFT (like Number, Vocabulary...)
# [4] feature type in TF
# [5] whether it is a Feature, Target or Infer

# close price before N days
FORMAT_INTEGER = "Integer"
FORMAT_NUMBER = "Number"
FORMAT_VOCABULARY = "Vocabulary"

TYPE_FEATURE = "feature"
TYPE_TARGET = "target"
TYPE_INFER = "infer"

### Features
feature_extractor_definition["time"] = ("str", "STRING", "tf.FixedLenFeature", FORMAT_VOCABULARY, tf.string, TYPE_FEATURE)
feature_extractor_definition["share_id"] = ("str", "STRING", "tf.FixedLenFeature", FORMAT_VOCABULARY, tf.string, TYPE_FEATURE)

for i in range(feature_definition_config["hloc_seq_step"]-1, 0, -1):
    feature_extractor_definition["close_b{}".format(i-1)] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)

for i in range(feature_definition_config["hloc_seq_step"] - 1, 0, -1):
    feature_extractor_definition["open_b{}".format(i-1)] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
for i in range(feature_definition_config["hloc_seq_step"] - 1, 0, -1):
    feature_extractor_definition["high_b{}".format(i-1)] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
for i in range(feature_definition_config["hloc_seq_step"] - 1, 0, -1):
    feature_extractor_definition["low_b{}".format(i-1)] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
# for i in range(feature_definition_config["hloc_seq_step"] - 1, 0, -1):
#     feature_extractor_definition["volume_b{}".format(i-1)] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)

### Targets
# only trading day, exclude holidays
# feature_extractor_definition["ror_1_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_TARGET)
# feature_extractor_definition["ror_5_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_TARGET)
# feature_extractor_definition["ror_10_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_TARGET)
# feature_extractor_definition["ror_20_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_TARGET)
# feature_extractor_definition["ror_40_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_TARGET)
# feature_extractor_definition["ror_60_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_TARGET)

# for sequence
hloc = ['high','low','open','close']
for i in hloc:
    feature_extractor_definition["target_{}_price".format(i)] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_TARGET)
for i in hloc:
    feature_extractor_definition["target_{}_trend".format(i)] = ("int", "REAL", "tf.FixedLenFeature", FORMAT_INTEGER, tf.int32, TYPE_TARGET)

# feature_extractor_definition["volume"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
# feature_extractor_definition["target_volume"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_TARGET)

### Infer
# This type if infered from coder which is only exist in features or targets, we don't need to store it in db.
# feature_extractor_definition["ror_1_days_beyond_0_001_bool"] = ("int", "INT", "tf.FixedLenFeature", FORMAT_INTEGER, tf.int64, TYPE_INFER)
# feature_extractor_definition["ror_1_days_beyond_0_001_bool"] = ("float", "INT", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_INFER)
# feature_extractor_definition["ror_20_days_bool"] = ("int", "INT", "tf.FixedLenFeature", FORMAT_INTEGER, tf.int64, TYPE_INFER)

