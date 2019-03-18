import tensorflow as tf

feature_definition_config = {}
feature_definition_config["close_n_days_before"] = 21
feature_definition_config["ror_n_days_after"] = 60

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
# [5] whether it is a Feature or Target
# [6] whether it is enable

# close price before N days
FORMAT_NUMBER = "Number"
FORMAT_VOCABULARY = "Vocabulary"

TYPE_FEATURE = "feature"
TYPE_TARGET = "target"

### Features
feature_extractor_definition["close_b0"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b1"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b2"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b3"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b4"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b5"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b6"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b7"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b8"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b9"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b10"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b11"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b12"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b13"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b14"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b15"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b16"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b17"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b18"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b19"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["close_b20"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE, True)
feature_extractor_definition["share_id"] = ("str", "REAL", "tf.FixedLenFeature", FORMAT_VOCABULARY, tf.string, TYPE_FEATURE, True)

### Targets
## only trading day, exclude holidays
feature_extractor_definition["ror_05_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_TARGET, False)
feature_extractor_definition["ror_10_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_TARGET, False)
feature_extractor_definition["ror_20_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_TARGET, False)
feature_extractor_definition["ror_40_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_TARGET, False)
feature_extractor_definition["ror_60_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_TARGET, False)

# feature_column_definition is used for preprocessing and training, it is include the feature added dynamically
# all items in `feature_column_definition` are not stored in database
# TARGET_KEY_ROR_20_DATS_BOOL = 'ror_20_days_bool'
#
# feature_column_definition = {}
# for key, value in feature_extractor_definition.iteritems():
#     if value[6]:
#         feature_column_definition[key] = value
# feature_column_definition[TARGET_KEY_ROR_20_DATS_BOOL] = ("bool", "None", "tf.FixedLenFeature", FORMAT_NUMBER, tf.int64, TYPE_TARGET, True)


# def ror_20_days_bool_function(header, row):
#     index = header.index('ror_20_days')
#     return 1 if float(row[index]) > 0 else 0


# # feature columns added dynamically
# new_feature_column_names = [TARGET_KEY_ROR_20_DATS_BOOL]
# # feature columns decider
# new_feature_column_functions = [ror_20_days_bool_function]
#
# # types for TFT
# enabled_number_features = []
# enabled_vocabulary_features = []
# for key, value in feature_extractor_definition.iteritems():
#     if value[6]:  # if enabled in config
#         if value[3] == FORMAT_NUMBER:
#             enabled_number_features.append(key)
#         elif value[3] == FORMAT_VOCABULARY:
#             enabled_vocabulary_features.append(key)
#         else:
#             raise Exception("unsupported feature types in TFT")
#
#
# # todo: what about `time`?
# # todo: move to DataFormatter for the specific algorithm
# enabled_feature_keys = []
# enabled_target_keys = []
#
# for key,value in feature_column_definition.iteritems():
#     if value[6]:
#         if value[5] == TYPE_TARGET:
#             enabled_target_keys.append(key)
#         elif value[5] == TYPE_FEATURE:
#             enabled_feature_keys.append(key)
#         else:
#             raise Exception("Unsupported feature type:" + str(value[5]))

# python type to tf type dict
TYPE_DICT = {
    'str': tf.string,  # String and bool features
    'int': tf.int64,  # Int features that represent numeric values
    'float': tf.float32,  # Floats
}