"""
Binary for trianing a RNN-based classifier for the Quick, Draw! data.
python train_model.py \
    --training_data train_data \
    --eval_data eval_data \
    --model_dir /tmp/quickdraw_model/ \
    --cell_type cudnn_lstm
When running on GPUs using --cell_type cudnn_lstm is much faster.
The expected performance is ~75% in 1.5M steps with the default configuration.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import ast
import functools
import sys

import tensorflow as tf

from model.rnn import get_input_fn, model_fn


def get_num_classes():
    classes = []
    with tf.gfile.GFile(FLAGS.classes_file, "r") as f:
        classes = [x for x in f]
    num_classes = len(classes)
    return num_classes


def create_estimator_and_specs(run_config):
    """Creates an Experiment configuration based on the estimator and input fn."""
    model_params = tf.contrib.training.HParams(
        num_layers=FLAGS.num_layers,
        num_nodes=FLAGS.num_nodes,
        batch_size=FLAGS.batch_size,
        num_conv=ast.literal_eval(FLAGS.num_conv),
        conv_len=ast.literal_eval(FLAGS.conv_len),
        num_classes=get_num_classes(),
        learning_rate=FLAGS.learning_rate,
        gradient_clipping_norm=FLAGS.gradient_clipping_norm,
        cell_type=FLAGS.cell_type,
        batch_norm=FLAGS.batch_norm,
        dropout=FLAGS.dropout)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=model_params)

    train_spec = tf.estimator.TrainSpec(input_fn=get_input_fn(
        mode=tf.estimator.ModeKeys.TRAIN,
        tfrecord_pattern=FLAGS.training_data,
        batch_size=FLAGS.batch_size), max_steps=FLAGS.steps)

    eval_spec = tf.estimator.EvalSpec(input_fn=get_input_fn(
        mode=tf.estimator.ModeKeys.EVAL,
        tfrecord_pattern=FLAGS.eval_data,
        batch_size=FLAGS.batch_size))

    return estimator, train_spec, eval_spec


def main(unused_args):
    estimator, train_spec, eval_spec = create_estimator_and_specs(
        run_config=tf.estimator.RunConfig(
            model_dir=FLAGS.model_dir,
            save_checkpoints_secs=300,
            save_summary_steps=100))
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--training_data",
        type=str,
        default="",
        help="Path to training data (tf.Example in TFRecord format)")
    parser.add_argument(
        "--eval_data",
        type=str,
        default="",
        help="Path to evaluation data (tf.Example in TFRecord format)")
    parser.add_argument(
        "--classes_file",
        type=str,
        default="",
        help="Path to a file with the classes - one class per line")
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of recurrent neural network layers.")
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=128,
        help="Number of node per recurrent network layer.")
    parser.add_argument(
        "--num_conv",
        type=str,
        default="[48, 64, 96]",
        help="Number of conv layers along with number of filters per layer.")
    parser.add_argument(
        "--conv_len",
        type=str,
        default="[5, 5, 3]",
        help="Length of the convolution filters.")
    parser.add_argument(
        "--cell_type",
        type=str,
        default="lstm",
        help="Cell type used for rnn layers: cudnn_lstm, lstm or block_lstm.")
    parser.add_argument(
        "--batch_norm",
        type="bool",
        default="False",
        help="Whether to enable batch normalization or not.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate used for training.")
    parser.add_argument(
        "--gradient_clipping_norm",
        type=float,
        default=9.0,
        help="Gradient clipping norm used during training.")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout used for convolutions and bidi lstm layers.")
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Number of training steps.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size to use for training/evaluation.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Path for storing the model checkpoints.")
    parser.add_argument(
        "--self_test",
        type="bool",
        default="False",
        help="Whether to enable batch normalization or not.")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
