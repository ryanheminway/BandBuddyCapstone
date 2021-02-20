import tensorflow.compat.v1 as tf


# UTILITY FUNCTION NORMALLY IN MAGENTA.COMMON
def flatten_maybe_padded_sequences(maybe_padded_sequences, lengths=None):
  """Flattens the batch of sequences, removing padding (if applicable).

  Args:
    maybe_padded_sequences: A tensor of possibly padded sequences to flatten,
        sized `[N, M, ...]` where M = max(lengths).
    lengths: Optional length of each sequence, sized `[N]`. If None, assumes no
        padding.

  Returns:
     flatten_maybe_padded_sequences: The flattened sequence tensor, sized
         `[sum(lengths), ...]`.
  """
  def flatten_unpadded_sequences():
    # The sequences are equal length, so we should just flatten over the first
    # two dimensions.
    return tf.reshape(maybe_padded_sequences,
                      [-1] + maybe_padded_sequences.shape.as_list()[2:])

  if lengths is None:
    return flatten_unpadded_sequences()

  def flatten_padded_sequences():
    indices = tf.where(tf.sequence_mask(lengths))
    return tf.gather_nd(maybe_padded_sequences, indices)

  return tf.cond(
      tf.equal(tf.reduce_min(lengths), tf.shape(maybe_padded_sequences)[1]),
      flatten_unpadded_sequences,
      flatten_padded_sequences)
