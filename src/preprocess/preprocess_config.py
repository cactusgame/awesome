SHUF_MEM = 1  # the shuf command memory usage Gi

TRAIN_SPLIT_RATIO = 0.8 # the ratio for split train and eval data set

# Split the train and eval sets into `DATASET_NUM_SHARDS` shards. Allows for parallel
# preprocessing and is used for shuffling the dataset.
DATASET_NUM_SHARDS = 2