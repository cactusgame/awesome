import tensorflow as tf

feature_definition_config = {}
feature_definition_config["close_n_days_before"] = 3
feature_definition_config["ror_n_days_after"] = 60

feature_definition = {}
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

feature_definition["close_b0"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32)
feature_definition["close_b1"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32)
feature_definition["close_b2"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32)
feature_definition["share_id"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_VOCABULARY, tf.string)

# RoR (rate of return)
## only trading day, exclude holidays
feature_definition["ror_05_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32)
feature_definition["ror_10_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32)
feature_definition["ror_20_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32)
feature_definition["ror_40_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32)
feature_definition["ror_60_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32)

# types for TFT
number_keys = []
vocabulary_keys = []
for key, value in feature_definition.iteritems():
    if value[3] == FORMAT_NUMBER:
        number_keys.append(key)
    elif value[3] == FORMAT_VOCABULARY:
        vocabulary_keys.append(key)
    else:
        raise Exception("unsupported feature types in TFT")
