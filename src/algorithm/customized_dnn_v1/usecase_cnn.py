import numpy as np
import os
import re
import shutil
import sys
import tarfile
import tensorflow as tf
from six.moves import urllib

print(tf.__version__)

DATA_DIR = 'data'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def maybe_download_and_extract(
        dest_directory='data',
        data_url='http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'):
    """Download and extract the tarball from Alex's website."""
    dest_directory = dest_directory
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

maybe_download_and_extract(DATA_DIR, DATA_URL)


def extract_data(index=0, filepath='data/cifar-10-batches-bin/data_batch_5.bin'):
    bytestream = open(filepath, mode='rb')

    label_bytes_length = 1
    image_bytes_length = (32 ** 2) * 3
    record_bytes_length = label_bytes_length + image_bytes_length

    bytestream.seek(record_bytes_length * index, 0)
    label_bytes = bytestream.read(label_bytes_length)
    image_bytes = bytestream.read(image_bytes_length)

    label = np.frombuffer(label_bytes, dtype=np.uint8)
    image = np.frombuffer(image_bytes, dtype=np.uint8)

    image = np.reshape(image, [3, 32, 32])
    image = np.transpose(image, [1, 2, 0])
    image = image.astype(np.float32)

    result = {
        'image': image,
        'label': label,
    }
    bytestream.close()
    return result

result = extract_data(np.random.randint(1000))

class FLAGS():
  pass

FLAGS.batch_size = 200
FLAGS.max_steps = 1000
FLAGS.eval_steps = 100
FLAGS.save_checkpoints_steps = 100
FLAGS.tf_random_seed = 19851211
FLAGS.model_name = 'cnn-model-02'
FLAGS.use_checkpoint = False


IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3
NUM_CLASSES = 10


def parse_record(raw_record):
    # Every record consists of a label followed by the image, with a fixed number
    # of bytes for each.
    label_bytes = 1
    image_bytes = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH
    record_bytes = label_bytes + image_bytes

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_vector = tf.decode_raw(raw_record, tf.uint8)

    # The first byte represents the label, which we convert from uint8 to int32
    # and then to one-hot.
    label = tf.cast(record_vector[0], tf.int32)
    label = tf.one_hot(label, NUM_CLASSES)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        record_vector[label_bytes:record_bytes], [IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH])

    # Convert from [depth, height, width] to [height, width, depth], and cast as
    # float32.
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    return image, label


def preprocess_image(image, is_training=False):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, IMAGE_HEIGHT + 8, IMAGE_WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


def generate_input_fn(file_names, mode=tf.estimator.ModeKeys.EVAL, batch_size=1):
    def _input_fn():
        label_bytes = 1
        image_bytes = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH
        record_bytes = label_bytes + image_bytes
        dataset = tf.data.FixedLengthRecordDataset(filenames=file_names,
                                                   record_bytes=record_bytes)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if is_training:
            buffer_size = batch_size * 2 + 1
            dataset = dataset.shuffle(buffer_size=buffer_size)

        dataset = dataset.map(parse_record)
        dataset = dataset.map(lambda image, label: (preprocess_image(image, is_training), label))

        # We call repeat after shuffling, rather than before, to prevent separate
        # epochs from blending together.
        dataset = dataset.repeat()
        dataset = dataset.prefetch(2 * batch_size)

        # Batch results by up to batch_size, and then fetch the tuple from the
        # iterator.
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()

        features = {'images': images}
        return features, labels

    return _input_fn


def get_feature_columns():
  feature_columns = {
    'images': tf.feature_column.numeric_column('images', (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)),
  }
  return feature_columns

feature_columns = get_feature_columns()
print("Feature Columns: {}".format(feature_columns))


def inference(images):
  # 1st Convolutional Layer
  conv1 = tf.layers.conv2d(
      inputs=images, filters=64, kernel_size=[5, 5], padding='same',
      activation=tf.nn.relu, name='conv1')
  pool1 = tf.layers.max_pooling2d(
      inputs=conv1, pool_size=[3, 3], strides=2, name='pool1')
  norm1 = tf.nn.lrn(
      pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

  # 2nd Convolutional Layer
  conv2 = tf.layers.conv2d(
      inputs=norm1, filters=64, kernel_size=[5, 5], padding='same',
      activation=tf.nn.relu, name='conv2')
  norm2 = tf.nn.lrn(
      conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
  pool2 = tf.layers.max_pooling2d(
      inputs=norm2, pool_size=[3, 3], strides=2, name='pool2')

  # Flatten Layer
  shape = pool2.get_shape()
  pool2_ = tf.reshape(pool2, [-1, shape[1]*shape[2]*shape[3]])

  # 1st Fully Connected Layer
  dense1 = tf.layers.dense(
      inputs=pool2_, units=384, activation=tf.nn.relu, name='dense1')

  # 2nd Fully Connected Layer
  dense2 = tf.layers.dense(
      inputs=dense1, units=192, activation=tf.nn.relu, name='dense2')

  # 3rd Fully Connected Layer (Logits)
  logits = tf.layers.dense(
      inputs=dense2, units=NUM_CLASSES, activation=tf.nn.relu, name='logits')

  return logits


def model_fn(features, labels, mode, params):
  # Create the input layers from the features
  feature_columns = list(get_feature_columns().values())

  images = tf.feature_column.input_layer(
    features=features, feature_columns=feature_columns)

  images = tf.reshape(
    images, shape=(-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))

  # Calculate logits through CNN
  logits = inference(images)

  if mode in (tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL):
    predicted_indices = tf.argmax(input=logits, axis=1)
    probabilities = tf.nn.softmax(logits, name='softmax_tensor')

  if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
    global_step = tf.train.get_or_create_global_step()
    label_indices = tf.argmax(input=labels, axis=1)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=labels, logits=logits)
    tf.summary.scalar('cross_entropy', loss)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'classes': predicted_indices,
        'probabilities': probabilities
    }
    export_outputs = {
        'predictions': tf.estimator.export.PredictOutput(predictions)
    }
    return tf.estimator.EstimatorSpec(
        mode, predictions=predictions, export_outputs=export_outputs)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, train_op=train_op)

  if mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)
    }
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=eval_metric_ops)


def serving_input_fn():
    receiver_tensor = {'images': tf.placeholder(
        shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=tf.float32)}
    features = {'images': tf.map_fn(preprocess_image, receiver_tensor['images'])}

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)








# =======================================================
# =======================================================
# =======================================================

model_dir = 'trained_models/{}'.format(FLAGS.model_name)
train_data_files = ['data/cifar-10-batches-bin/data_batch_{}.bin'.format(i) for i in range(1,5)]
valid_data_files = ['data/cifar-10-batches-bin/data_batch_5.bin']
test_data_files = ['data/cifar-10-batches-bin/test_batch.bin']


run_config = tf.estimator.RunConfig(
  save_checkpoints_steps=FLAGS.save_checkpoints_steps,
  tf_random_seed=FLAGS.tf_random_seed,
  model_dir=model_dir
)

estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

# There is another Exporter named FinalExporter
exporter = tf.estimator.LatestExporter(
  name='Servo',
  serving_input_receiver_fn=serving_input_fn,
  assets_extra=None,
  as_text=False,
  exports_to_keep=5)

train_spec = tf.estimator.TrainSpec(
  input_fn=generate_input_fn(file_names=train_data_files,
                             mode=tf.estimator.ModeKeys.TRAIN,
                             batch_size=FLAGS.batch_size),
  max_steps=FLAGS.max_steps)

eval_spec = tf.estimator.EvalSpec(
  input_fn=generate_input_fn(file_names=valid_data_files,
                             mode=tf.estimator.ModeKeys.EVAL,
                             batch_size=FLAGS.batch_size),
  steps=FLAGS.eval_steps, exporters=exporter)


if not FLAGS.use_checkpoint:
  print("Removing previous artifacts...")
  shutil.rmtree(model_dir, ignore_errors=True)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


test_input_fn = generate_input_fn(file_names=test_data_files,
                                  mode=tf.estimator.ModeKeys.EVAL,
                                  batch_size=1000)
estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)
print(estimator.evaluate(input_fn=test_input_fn, steps=1))


export_dir = model_dir + '/export/Servo/'
saved_model_dir = os.path.join(export_dir, os.listdir(export_dir)[-1])

predictor_fn = tf.contrib.predictor.from_saved_model(
  export_dir = saved_model_dir,
  signature_def_key='predictions')

N = 1000
labels = []
images = []

for i in range(N):
  result = extract_data(i, filepath='data/cifar-10-batches-bin/test_batch.bin')
  images.append(result['image'])
  labels.append(result['label'][0])

output = predictor_fn({'images': images})


np.sum([a==r for a, r in zip(labels, output['classes'])]) / float(N)