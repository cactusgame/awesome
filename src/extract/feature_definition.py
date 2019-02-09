import tensorflow as tf

feature_definition_config = {}
feature_definition_config["close_n_days_before"] = 3
feature_definition_config["ror_n_days_after"] = 60

# pick up all features we will use
valid_feature_column_definition = []
valid_feature_column_definition += ["close_b0", "close_b1", "close_b2"]
valid_feature_column_definition += ["share_id"]
valid_feature_column_definition += ["ror_20_days","ror_20_days_bool"]

# FEATURE_KEYS
FEATURE_KEYS = []
FEATURE_KEYS += ["close_b0", "close_b1", "close_b2"]
FEATURE_KEYS += ["share_id"]

# TARGET_KEYS
TARGET_KEYS = []
TARGET_KEYS += ["ror_05_days","ror_10_days","ror_20_days","ror_40_days","ror_60_days","ror_20_days_bool"]



# feature_extractor_definition is only used for feature extractor,
# items in the `feature_extractor_definition` will create a column in database
feature_extractor_definition = {}
# format:
# key: feature name
# value:
# [0] type in python
# [1] type in sqlite
# [2] feature def in TFT
# [3] feature preprocessing type in TFT (like Number, Vocabulary...)
# [4] feature type in TF

# close price before N days
FORMAT_NUMBER = "Number"
FORMAT_VOCABULARY = "Vocabulary"

feature_extractor_definition["close_b0"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32)
feature_extractor_definition["close_b1"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32)
feature_extractor_definition["close_b2"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32)
feature_extractor_definition["share_id"] = ("str", "REAL", "tf.FixedLenFeature", FORMAT_VOCABULARY, tf.string)

# RoR (rate of return)
## only trading day, exclude holidays
feature_extractor_definition["ror_05_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32)
feature_extractor_definition["ror_10_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32)
feature_extractor_definition["ror_20_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32)
feature_extractor_definition["ror_40_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32)
feature_extractor_definition["ror_60_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32)

# feature_column_definition is used for preprocessing and training, it is include the feature added dynamically
# all items in `feature_column_definition` are not stored in database
TARGET_KEY_ROR_20_DATS_BOOL = 'ror_20_days_bool'

feature_column_definition = {}
for key, value in feature_extractor_definition.iteritems():
    if key in valid_feature_column_definition:
        feature_column_definition[key] = value
feature_column_definition[TARGET_KEY_ROR_20_DATS_BOOL] = ("bool", "None", "tf.FixedLenFeature", FORMAT_NUMBER, tf.int64)


def ror_20_days_bool_function(header, row):
    index = header.index('ror_20_days')
    return 1 if float(row[index]) > 0 else 0


# feature columns added dynamically
new_feature_column_names = [TARGET_KEY_ROR_20_DATS_BOOL]
# feature columns decider
new_feature_column_functions = [ror_20_days_bool_function]

# types for TFT
number_keys = []
vocabulary_keys = []
for key, value in feature_extractor_definition.iteritems():
    if key in valid_feature_column_definition:
        if value[3] == FORMAT_NUMBER:
            number_keys.append(key)
        elif value[3] == FORMAT_VOCABULARY:
            vocabulary_keys.append(key)
        else:
            raise Exception("unsupported feature types in TFT")


# todo: distingwish feature,target, enable,disable