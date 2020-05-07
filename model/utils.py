import tensorflow as tf


def _parse_tfexample_fn(example_proto, mode):
    """Parse a single record which is expected to be a tensorflow.Example."""
    feature_to_type = {
        "ink": tf.VarLenFeature(dtype=tf.float32),
        "shape": tf.FixedLenFeature([2], dtype=tf.int64)
    }
    if mode != tf.estimator.ModeKeys.PREDICT:
        # The labels won't be available at inference time, so don't add them
        # to the list of feature_columns to be read.
        feature_to_type["class_index"] = tf.FixedLenFeature(
            [1], dtype=tf.int64)

    parsed_features = tf.parse_single_example(
        example_proto, feature_to_type)
    labels = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = parsed_features["class_index"]
    parsed_features["ink"] = tf.sparse_tensor_to_dense(
        parsed_features["ink"])
    return parsed_features, labels
