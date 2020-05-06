import tensorflow as tf
import functools

from . utils import _parse_tfexample_fn


def get_input_fn(mode, tfrecord_pattern, batch_size):
    """
    Creates an input_fn that stores all the data in memory.
    Args:
        mode: one of tf.contrib.learn.ModeKeys.{TRAIN, INFER, EVAL}
        tfrecord_pattern: path to a TF record file created using create_dataset.py.
        batch_size: the batch size to output.
    Returns:
        A valid input_fn for the model estimator.
    """

    def _input_fn():
        """
        Estimator `input_fn`.
        Returns:
          A tuple of:
          - Dictionary of string feature name to `Tensor`.
          - `Tensor` of target labels.
        """
        dataset = tf.data.TFRecordDataset.list_files(tfrecord_pattern)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=10)
        dataset = dataset.repeat()
        # Preprocesses 10 files concurrently and interleaves records from each file.
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=10,
            block_length=1)
        dataset = dataset.map(
            functools.partial(_parse_tfexample_fn, mode=mode),
            num_parallel_calls=10)
        dataset = dataset.prefetch(10000)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=1000000)
        # Our inputs are variable length, so pad them.
        dataset = dataset.padded_batch(
            batch_size, padded_shapes=dataset.output_shapes)
        features, labels = dataset.make_one_shot_iterator().get_next()
        return features, labels

    return _input_fn


def model_fn(features, labels, mode, params):
    """
    Model function for RNN classifier.
    This function sets up a neural network which applies convolutional layers (as
    configured with params.num_conv and params.conv_len) to the input.
    The output of the convolutional layers is given to LSTM layers (as configured
    with params.num_layers and params.num_nodes).
    The final state of the all LSTM layers are concatenated and fed to a fully
    connected layer to obtain the final classification scores.
    Args:
        features: dictionary with keys: inks, lengths.
        labels: one hot encoded classes
        mode: one of tf.estimator.ModeKeys.{TRAIN, INFER, EVAL}
        params: a parameter dictionary with the following keys: num_layers,
            num_nodes, batch_size, num_conv, conv_len, num_classes, learning_rate.
    Returns:
        ModelFnOps for Estimator API.
    """

    def _get_input_tensors(features, labels):
        """Converts the input dict into inks, lengths, and labels tensors."""
        # features[ink] is a sparse tensor that is [8, batch_maxlen, 3]
        # inks will be a dense tensor of [8, maxlen, 3]
        # shapes is [batchsize, 2]
        shapes = features["shape"]
        # lengths will be [batch_size]
        lengths = tf.squeeze(
            tf.slice(shapes, begin=[0, 0], size=[params.batch_size, 1]))
        inks = tf.reshape(features["ink"], [params.batch_size, -1, 3])
        if labels is not None:
            labels = tf.squeeze(labels)
        return inks, lengths, labels

    def _add_conv_layers(inks, lengths):
        """Adds convolution layers."""
        convolved = inks
        for i in range(len(params.num_conv)):
            convolved_input = convolved
            if params.batch_norm:
                convolved_input = tf.layers.batch_normalization(
                    convolved_input,
                    training=(mode == tf.estimator.ModeKeys.TRAIN))
            # Add dropout layer if enabled and not first convolution layer.
            if i > 0 and params.dropout:
                convolved_input = tf.layers.dropout(
                    convolved_input,
                    rate=params.dropout,
                    training=(mode == tf.estimator.ModeKeys.TRAIN))
            convolved = tf.layers.conv1d(
                convolved_input,
                filters=params.num_conv[i],
                kernel_size=params.conv_len[i],
                activation=None,
                strides=1,
                padding="same",
                name="conv1d_%d" % i)
        return convolved, lengths

    def _add_regular_rnn_layers(convolved, lengths):
        """Adds RNN layers."""
        if params.cell_type == "lstm":
            cell = tf.nn.rnn_cell.BasicLSTMCell
        elif params.cell_type == "block_lstm":
            cell = tf.contrib.rnn.LSTMBlockCell
        cells_fw = [cell(params.num_nodes) for _ in range(params.num_layers)]
        cells_bw = [cell(params.num_nodes) for _ in range(params.num_layers)]
        if params.dropout > 0.0:
            cells_fw = [tf.contrib.rnn.DropoutWrapper(
                cell) for cell in cells_fw]
            cells_bw = [tf.contrib.rnn.DropoutWrapper(
                cell) for cell in cells_bw]
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=cells_fw,
            cells_bw=cells_bw,
            inputs=convolved,
            sequence_length=lengths,
            dtype=tf.float32,
            scope="rnn_classification")
        return outputs

    def _add_cudnn_rnn_layers(convolved):
        """Adds CUDNN LSTM layers."""
        # Convolutions output [B, L, Ch], while CudnnLSTM is time-major.
        convolved = tf.transpose(convolved, [1, 0, 2])
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=params.num_layers,
            num_units=params.num_nodes,
            dropout=params.dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0,
            direction="bidirectional")
        outputs, _ = lstm(convolved)
        # Convert back from time-major outputs to batch-major outputs.
        outputs = tf.transpose(outputs, [1, 0, 2])
        return outputs

    def _add_rnn_layers(convolved, lengths):
        """Adds recurrent neural network layers depending on the cell type."""
        if params.cell_type != "cudnn_lstm":
            outputs = _add_regular_rnn_layers(convolved, lengths)
        else:
            outputs = _add_cudnn_rnn_layers(convolved)
        # outputs is [batch_size, L, N] where L is the maximal sequence length and N
        # the number of nodes in the last layer.
        mask = tf.tile(
            tf.expand_dims(tf.sequence_mask(lengths, tf.shape(outputs)[1]), 2),
            [1, 1, tf.shape(outputs)[2]])
        zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))
        outputs = tf.reduce_sum(zero_outside, axis=1)
        return outputs

    def _add_fc_layers(final_state):
        """Adds a fully connected layer."""
        return tf.layers.dense(final_state, params.num_classes)

    # Build the model.
    inks, lengths, labels = _get_input_tensors(features, labels)
    convolved, lengths = _add_conv_layers(inks, lengths)
    final_state = _add_rnn_layers(convolved, lengths)
    logits = _add_fc_layers(final_state)
    # Add the loss.
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits))
    # Add the optimizer.
    train_op = tf.contrib.layers.optimize_loss(
        loss=cross_entropy,
        global_step=tf.train.get_global_step(),
        learning_rate=params.learning_rate,
        optimizer="Adam",
        # some gradient clipping stabilizes training in the beginning.
        clip_gradients=params.gradient_clipping_norm,
        summaries=["learning_rate", "loss", "gradients", "gradient_norm"])
    # Compute current predictions.
    predictions = tf.argmax(logits, axis=1)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"logits": logits, "predictions": predictions},
        loss=cross_entropy,
        train_op=train_op,
        eval_metric_ops={"accuracy": tf.metrics.accuracy(labels, predictions)})
