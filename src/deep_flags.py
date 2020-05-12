from __future__ import division
import os
from os.path import join as pjoin

import sys

import tensorflow as tf


# IMAGE_PIXELS = 1000
# NUM_CLASSES = 10
# directory = "Epinions-500"

def setup_flags(image_pixels, directory):
    with open(os.path.join(directory, 'data.csv')) as fin:
        num_data = 0
        for _ in fin:
            num_data += 1
    with open(os.path.join(directory, 'train.txt')) as fin:
        num_train = 0
        for _ in fin:
            num_train += 1
    with open(os.path.join(directory, 'test.txt')) as fin:
        num_test = 0
        for _ in fin:
            num_test += 1

    def home_out(path):
        return pjoin(os.environ['HOME'], 'tmp', 'mnist', path)

    flags = tf.app.flags

    # Autoencoder Architecture Specific Flags
    flags.DEFINE_integer("num_hidden_layers", 3, "Number of hidden layers")

    flags.DEFINE_integer('hidden1_units', 512,
                         'Number of units in hidden layer 1.')
    flags.DEFINE_integer('hidden2_units', 128,
                         'Number of units in hidden layer 2.')
    flags.DEFINE_integer('hidden3_units', 64,
                         'Number of units in hidden layer 3.')
    flags.DEFINE_integer('num_data', num_data, 'Number of classes')
    flags.DEFINE_integer('num_train', num_train, 'Number of classes')
    flags.DEFINE_integer('num_test', num_test, 'Number of classes')
    flags.DEFINE_integer('num_finetune', 5000, 'Number of classes')

    flags.DEFINE_integer('image_pixels', image_pixels, 'Total number of pixels')
    flags.DEFINE_integer('num_classes', 2, 'Number of classes')

    flags.DEFINE_float('pre_layer1_learning_rate', 0.001,
                       'Initial learning rate.')
    flags.DEFINE_float('pre_layer2_learning_rate', 0.001,
                       'Initial learning rate.')
    flags.DEFINE_float('pre_layer3_learning_rate', 0.001,
                       'Initial learning rate.')

    flags.DEFINE_float('noise_1', 0.50, 'Rate at which to set pixels to 0')
    flags.DEFINE_float('noise_2', 0.50, 'Rate at which to set pixels to 0')
    flags.DEFINE_float('noise_3', 0.50, 'Rate at which to set pixels to 0')

    # Constants
    flags.DEFINE_integer('seed', 1234, 'Random seed')
    flags.DEFINE_integer('image_size', 28, 'Image square size')

    flags.DEFINE_integer('batch_size', 1000,
                         'Batch size. Must divide evenly into the dataset sizes.')

    flags.DEFINE_float('finetune_learning_rate', 0.001,
                       'Supervised initial learning rate.')
    flags.DEFINE_float('alpha', 8, 'alpha')

    flags.DEFINE_integer('pretraining_epochs', 10,
                         "Number of training epochs for pretraining layers")
    flags.DEFINE_integer('combine_epochs', 50,
                         "Number of training epochs for pretraining layers")
    flags.DEFINE_integer('finetuning_epochs', 200,
                         "Number of training epochs for "
                         "fine tuning supervised step")

    flags.DEFINE_float('zero_bound', 1.0e-9,
                       'Value to use as buffer to avoid '
                       'numerical issues at 0')
    flags.DEFINE_float('one_bound', 1.0 - 1.0e-9,
                       'Value to use as buffer to avoid numerical issues at 1')

    flags.DEFINE_float('flush_secs', 120, 'Number of seconds to flush summaries')

    # Directories
    flags.DEFINE_string('data_dir', home_out('data'),
                        'Directory to put the training data.')

    flags.DEFINE_string('summary_dir', home_out('summaries'),
                        'Directory to put the summary data')

    flags.DEFINE_string('chkpt_dir', home_out('chkpts'),
                        'Directory to put the model checkpoints')
    flags.DEFINE_string('directory', directory,
                        'Directory to put the model checkpoints')
    # TensorBoard
    flags.DEFINE_boolean('no_browser', True,
                         'Whether to start browser for TensorBoard')

    # Python
    flags.DEFINE_string('python', sys.executable,
                        'Path to python executable')

    return flags.FLAGS
