# Copyright 2021 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""LSTM-based encoders and decoders for MusicVAE."""
import abc

from magenta.common import flatten_maybe_padded_sequences
from magenta.common import Nade
import magenta.contrib.rnn as contrib_rnn
import magenta.contrib.seq2seq as contrib_seq2seq
import magenta.contrib.training as contrib_training
import base_model
import lstm_utils
import numpy as np
import tensorflow.compat.v1 as tf_one
import tensorflow_probability as tfp
import tensorflow.keras as tf_keras
# (NOTE Ryan Heminway)
# Tried to remove everything that is unused by our GrooVAE model

# ENCODERS
class LstmEncoder(base_model.BaseEncoder):
  """Unidirectional LSTM Encoder."""

  @property
  def output_depth(self):
    return self._cell.output_size

  def build(self, hparams, is_training=True, name_or_scope='encoder'):
    if hparams.use_cudnn:
      tf_one.logging.warning('cuDNN LSTM no longer supported. Using regular LSTM.')

    self._is_training = is_training
    self._name_or_scope = name_or_scope

    tf_one.logging.info('\nEncoder Cells (unidirectional):\n'
                    '  units: %s\n',
                        hparams.enc_rnn_size)
    self._cell = lstm_utils.rnn_cell(
        hparams.enc_rnn_size, hparams.dropout_keep_prob,
        hparams.residual_encoder, is_training)

  def encode(self, sequence, sequence_length):
    # Convert to time-major.
    sequence = tf_one.transpose(sequence, [1, 0, 2])
    outputs, _ = tf_one.nn.dynamic_rnn(
        self._cell, sequence, sequence_length, dtype=tf_one.float32,
        time_major=True, scope=self._name_or_scope)
    return outputs[-1]


class BidirectionalLstmEncoder(base_model.BaseEncoder):
  """Bidirectional LSTM Encoder."""

  @property
  def output_depth(self):
    return self._cells[0][-1].output_size + self._cells[1][-1].output_size

  def build(self, hparams, is_training=True, name_or_scope='encoder'):
    self._is_training = is_training
    self._name_or_scope = name_or_scope
    if hparams.use_cudnn:
      tf_one.logging.warning('cuDNN LSTM no longer supported. Using regular LSTM.')

    tf_one.logging.info('\nEncoder Cells (bidirectional):\n'
                    '  units: %s\n',
                        hparams.enc_rnn_size)

    self._cells = lstm_utils.build_bidirectional_lstm(
        layer_sizes=hparams.enc_rnn_size,
        dropout_keep_prob=hparams.dropout_keep_prob,
        residual=hparams.residual_encoder,
        is_training=is_training)

  def encode(self, sequence, sequence_length):
    cells_fw, cells_bw = self._cells

    output, states_fw, states_bw = contrib_rnn.stack_bidirectional_dynamic_rnn(
        cells_fw,
        cells_bw,
        sequence,
        sequence_length=sequence_length,
        time_major=False,
        dtype=tf_one.float32,
        scope=self._name_or_scope)
    # Note we access the outputs (h) from the states since the backward
    # outputs are reversed to the input order in the returned outputs.

    # (NOTE Ryan Heminway) changed syntax for accessing outputs based on change to LSTM cell definition
    #   Keras LSTMCell states do not have a .h property
    #last_h_fw = states_fw[-1][-1].h
    #last_h_bw = states_bw[-1][-1].h
    last_h_fw = states_fw[-1][-1][-1]
    last_h_bw = states_bw[-1][-1][-1]

    return tf_one.concat([last_h_fw, last_h_bw], 1)


class HierarchicalLstmEncoder(base_model.BaseEncoder):
  """Hierarchical LSTM encoder wrapper.

  Input sequences will be split into segments based on the first value of
  `level_lengths` and encoded. At subsequent levels, the embeddings will be
  grouped based on `level_lengths` and encoded until a single embedding is
  produced.

  See the `encode` method for details on the expected arrangement the sequence
  tensors.

  Args:
    core_encoder_cls: A single BaseEncoder class to use for each level of the
      hierarchy.
    level_lengths: A list of the (maximum) lengths of the segments at each
      level of the hierarchy. The product must equal `hparams.max_seq_len`.
  """

  def __init__(self, core_encoder_cls, level_lengths):
    self._core_encoder_cls = core_encoder_cls
    self._level_lengths = level_lengths

  @property
  def output_depth(self):
    return self._hierarchical_encoders[-1][1].output_depth

  @property
  def level_lengths(self):
    return list(self._level_lengths)

  def level(self, l):
    """Returns the BaseEncoder at level `l`."""
    return self._hierarchical_encoders[l][1]

  def build(self, hparams, is_training=True):
    self._total_length = hparams.max_seq_len
    if self._total_length != np.prod(self._level_lengths):
      raise ValueError(
          'The product of the HierarchicalLstmEncoder level lengths (%d) must '
          'equal the padded input sequence length (%d).' % (
              np.prod(self._level_lengths), self._total_length))
    tf_one.logging.info('\nHierarchical Encoder:\n'
                    '  input length: %d\n'
                    '  level lengths: %s\n',
                        self._total_length,
                        self._level_lengths)
    self._hierarchical_encoders = []
    num_splits = int(np.prod(self._level_lengths))
    for i, l in enumerate(self._level_lengths):
      num_splits //= l
      tf_one.logging.info('Level %d splits: %d', i, num_splits)
      h_encoder = self._core_encoder_cls()
      h_encoder.build(
          hparams, is_training,
          name_or_scope=tf_one.VariableScope(
              tf_one.AUTO_REUSE, 'encoder/hierarchical_level_%d' % i))
      self._hierarchical_encoders.append((num_splits, h_encoder))

  def encode(self, sequence, sequence_length):
    """Hierarchically encodes the input sequences, returning a single embedding.

    Each sequence should be padded per-segment. For example, a sequence with
    three segments [1, 2, 3], [4, 5], [6, 7, 8 ,9] and a `max_seq_len` of 12
    should be input as `sequence = [1, 2, 3, 0, 4, 5, 0, 0, 6, 7, 8, 9]` with
    `sequence_length = [3, 2, 4]`.

    Args:
      sequence: A batch of (padded) sequences, sized
        `[batch_size, max_seq_len, input_depth]`.
      sequence_length: A batch of sequence lengths. May be sized
        `[batch_size, level_lengths[0]]` or `[batch_size]`. If the latter,
        each length must either equal `max_seq_len` or 0. In this case, the
        segment lengths are assumed to be constant and the total length will be
        evenly divided amongst the segments.

    Returns:
      embedding: A batch of embeddings, sized `[batch_size, N]`.
    """
    batch_size = int(sequence.shape[0])
    sequence_length = lstm_utils.maybe_split_sequence_lengths(
        sequence_length, np.prod(self._level_lengths[1:]),
        self._total_length)

    for level, (num_splits, h_encoder) in enumerate(
        self._hierarchical_encoders):
      split_seqs = tf_one.split(sequence, num_splits, axis=1)
      # In the first level, we use the input `sequence_length`. After that,
      # we use the full embedding sequences.
      if level:
        sequence_length = tf_one.fill(
            [batch_size, num_splits], split_seqs[0].shape[1])
      split_lengths = tf_one.unstack(sequence_length, axis=1)
      embeddings = [
          h_encoder.encode(s, l) for s, l in zip(split_seqs, split_lengths)]
      sequence = tf_one.stack(embeddings, axis=1)

    with tf_one.control_dependencies([tf_one.assert_equal(tf_one.shape(sequence)[1], 1)]):
      return sequence[:, 0]


# DECODERS

class BaseLstmDecoder(base_model.BaseDecoder):
  """Abstract LSTM Decoder class.

  Implementations must define the following abstract methods:
      -`_sample`
      -`_flat_reconstruction_loss`
  """

  def build(self, hparams, output_depth, is_training=True):
    if hparams.use_cudnn:
      tf_one.logging.warning('cuDNN LSTM no longer supported. Using regular LSTM.')

    self._is_training = is_training

    tf_one.logging.info('\nDecoder Cells:\n'
                    '  units: %s\n',
                        hparams.dec_rnn_size)

    self._sampling_probability = lstm_utils.get_sampling_probability(
        hparams, is_training)
    self._output_depth = output_depth
    self._output_layer = tf_one.layers.Dense(
        output_depth, name='output_projection')
    self._dec_cell = lstm_utils.rnn_cell(
        hparams.dec_rnn_size, hparams.dropout_keep_prob,
        hparams.residual_decoder, is_training)

  @property
  def state_size(self):
    return self._dec_cell.state_size

  @abc.abstractmethod
  def _sample(self, rnn_output, temperature):
    """Core sampling method for a single time step.

    Args:
      rnn_output: The output from a single timestep of the RNN, sized
          `[batch_size, rnn_output_size]`.
      temperature: A scalar float specifying a sampling temperature.
    Returns:
      A batch of samples from the model.
    """
    pass

  @abc.abstractmethod
  def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
    """Core loss calculation method for flattened outputs.

    Args:
      flat_x_target: The flattened ground truth vectors, sized
        `[sum(x_length), self._output_depth]`.
      flat_rnn_output: The flattened output from all timeputs of the RNN,
        sized `[sum(x_length), rnn_output_size]`.
    Returns:
      r_loss: The unreduced reconstruction losses, sized `[sum(x_length)]`.
      metric_map: A map of metric names to tuples, each of which contain the
        pair of (value_tensor, update_op) from a tf.metrics streaming metric.
    """
    pass

  def _decode(self, z, helper, input_shape, max_length=None):
    """Decodes the given batch of latent vectors vectors, which may be 0-length.

    Args:
      z: Batch of latent vectors, sized `[batch_size, z_size]`, where `z_size`
        may be 0 for unconditioned decoding.
      helper: A seq2seq.Helper to use.
      input_shape: The shape of each model input vector passed to the decoder.
      max_length: (Optional) The maximum iterations to decode.

    Returns:
      results: The LstmDecodeResults.
    """
    initial_state = lstm_utils.initial_cell_state_from_embedding(
        self._dec_cell, z, name='decoder/z_to_initial_state')

    print("INITIAL STATE: ", initial_state)
    print("INPUT SHAPE: ", input_shape)

    decoder = lstm_utils.Seq2SeqLstmDecoder(
        self._dec_cell,
        helper,
        initial_state=initial_state,
        input_shape=input_shape,
        output_layer=self._output_layer)
    final_output, final_state, final_lengths = contrib_seq2seq.dynamic_decode(
        decoder,
        maximum_iterations=max_length,
        swap_memory=True,
        scope='decoder')
    results = lstm_utils.LstmDecodeResults(
        rnn_input=final_output.rnn_input[:, :, :self._output_depth],
        rnn_output=final_output.rnn_output,
        samples=final_output.sample_id,
        final_state=final_state,
        final_sequence_lengths=final_lengths)

    return results

  def reconstruction_loss(self, x_input, x_target, x_length, z=None,
                          c_input=None):
    """Reconstruction loss calculation.

    Args:
      x_input: Batch of decoder input sequences for teacher forcing, sized
        `[batch_size, max(x_length), output_depth]`.
      x_target: Batch of expected output sequences to compute loss against,
        sized `[batch_size, max(x_length), output_depth]`.
      x_length: Length of input/output sequences, sized `[batch_size]`.
      z: (Optional) Latent vectors. Required if model is conditional. Sized
        `[n, z_size]`.
      c_input: (Optional) Batch of control sequences, sized
          `[batch_size, max(x_length), control_depth]`. Required if conditioning
          on control sequences.

    Returns:
      r_loss: The reconstruction loss for each sequence in the batch.
      metric_map: Map from metric name to tf.metrics return values for logging.
      decode_results: The LstmDecodeResults.
    """
    batch_size = int(x_input.shape[0])

    has_z = z is not None
    z = tf_one.zeros([batch_size, 0]) if z is None else z
    repeated_z = tf_one.tile(
        tf_one.expand_dims(z, axis=1), [1, tf_one.shape(x_input)[1], 1])

    has_control = c_input is not None
    if c_input is None:
      c_input = tf_one.zeros([batch_size, tf_one.shape(x_input)[1], 0])

    sampling_probability_static = tf_one.get_static_value(
        self._sampling_probability)
    if sampling_probability_static == 0.0:
      # Use teacher forcing.
      x_input = tf_one.concat([x_input, repeated_z, c_input], axis=2)
      print("X INPUT FOR HELPER: ", x_input)
      helper = contrib_seq2seq.TrainingHelper(x_input, x_length)
    """else:
      # Use scheduled sampling.
      if has_z or has_control:
        auxiliary_inputs = tf.zeros([batch_size, tf.shape(x_input)[1], 0])
        if has_z:
          auxiliary_inputs = tf.concat([auxiliary_inputs, repeated_z], axis=2)
        if has_control:
          auxiliary_inputs = tf.concat([auxiliary_inputs, c_input], axis=2)
      else:
        auxiliary_inputs = None
      helper = contrib_seq2seq.ScheduledOutputTrainingHelper(
          inputs=x_input,
          sequence_length=x_length,
          auxiliary_inputs=auxiliary_inputs,
          sampling_probability=self._sampling_probability,
          next_inputs_fn=self._sample)"""

    decode_results = self._decode(
        z, helper=helper, input_shape=helper.inputs.shape[2:])
    print("DECODER INPUT SHAPE: ", helper.inputs.shape[2:])
    flat_x_target = flatten_maybe_padded_sequences(x_target, x_length)
    flat_rnn_output = flatten_maybe_padded_sequences(
        decode_results.rnn_output, x_length)
    r_loss, metric_map = self._flat_reconstruction_loss(
        flat_x_target, flat_rnn_output)

    # Sum loss over sequences.
    cum_x_len = tf_one.concat([(0,), tf_one.cumsum(x_length)], axis=0)
    r_losses = []
    for i in range(batch_size):
      b, e = cum_x_len[i], cum_x_len[i + 1]
      r_losses.append(tf_one.reduce_sum(r_loss[b:e]))
    r_loss = tf_one.stack(r_losses)

    return r_loss, metric_map, decode_results

  def sample(self, n, max_length=None, z=None, c_input=None, temperature=1.0,
             start_inputs=None, end_fn=None):
    """Sample from decoder with an optional conditional latent vector `z`.

    Args:
      n: Scalar number of samples to return.
      max_length: (Optional) Scalar maximum sample length to return. Required if
        data representation does not include end tokens.
      z: (Optional) Latent vectors to sample from. Required if model is
        conditional. Sized `[n, z_size]`.
      c_input: (Optional) Control sequence, sized `[max_length, control_depth]`.
      temperature: (Optional) The softmax temperature to use when sampling, if
        applicable.
      start_inputs: (Optional) Initial inputs to use for batch.
        Sized `[n, output_depth]`.
      end_fn: (Optional) A callable that takes a batch of samples (sized
        `[n, output_depth]` and emits a `bool` vector
        shaped `[batch_size]` indicating whether each sample is an end token.
    Returns:
      samples: Sampled sequences. Sized `[n, max_length, output_depth]`.
      final_state: The final states of the decoder.
    Raises:
      ValueError: If `z` is provided and its first dimension does not equal `n`.
    """
    if z is not None and int(z.shape[0]) != n:
      raise ValueError(
          '`z` must have a first dimension that equals `n` when given. '
          'Got: %d vs %d' % (z.shape[0], n))

    # Use a dummy Z in unconditional case.
    z = tf_one.zeros((n, 0), tf_one.float32) if z is None else z

    if c_input is not None:
      # Tile control sequence across samples.
      c_input = tf_one.tile(tf_one.expand_dims(c_input, 1), [1, n, 1])

    # If not given, start with zeros.
    if start_inputs is None:
      start_inputs = tf_one.zeros([n, self._output_depth], dtype=tf_one.float32)
    # In the conditional case, also concatenate the Z.
    start_inputs = tf_one.concat([start_inputs, z], axis=-1)
    if c_input is not None:
      start_inputs = tf_one.concat([start_inputs, c_input[0]], axis=-1)
    initialize_fn = lambda: (tf_one.zeros([n], tf_one.bool), start_inputs)

    sample_fn = lambda time, outputs, state: self._sample(outputs, temperature)
    end_fn = end_fn or (lambda x: False)

    def next_inputs_fn(time, outputs, state, sample_ids):
      del outputs
      finished = end_fn(sample_ids)
      next_inputs = tf_one.concat([sample_ids, z], axis=-1)
      if c_input is not None:
        # We need to stop if we've run out of control input.
        finished = tf_one.cond(tf_one.less(time, tf_one.shape(c_input)[0] - 1),
                               lambda: finished,
                               lambda: True)
        next_inputs = tf_one.concat([
            next_inputs,
            tf_one.cond(tf_one.less(time, tf_one.shape(c_input)[0] - 1),
                        lambda: c_input[time + 1],
                        lambda: tf_one.zeros_like(c_input[0]))  # should be unused
        ], axis=-1)
      return (finished, next_inputs, state)

    sampler = contrib_seq2seq.CustomHelper(
        initialize_fn=initialize_fn, sample_fn=sample_fn,
        next_inputs_fn=next_inputs_fn, sample_ids_shape=[self._output_depth],
        sample_ids_dtype=tf_one.float32)

    decode_results = self._decode(
        z, helper=sampler, input_shape=start_inputs.shape[1:],
        max_length=max_length)

    return decode_results.samples, decode_results

def get_default_hparams():
  """Returns copy of default HParams for LSTM models."""
  hparams_map = base_model.get_default_hparams().values()
  hparams_map.update({
      'conditional': True,
      'dec_rnn_size': [512],  # Decoder RNN: number of units per layer.
      'enc_rnn_size': [256],  # Encoder RNN: number of units per layer per dir.
      'dropout_keep_prob': 1.0,  # Probability all dropout keep.
      'sampling_schedule': 'constant',  # constant, exponential, inverse_sigmoid
      'sampling_rate': 0.0,  # Interpretation is based on `sampling_schedule`.
      'use_cudnn': False,  # DEPRECATED
      'residual_encoder': False,  # Use residual connections in encoder.
      'residual_decoder': False,  # Use residual connections in decoder.
      'control_preprocessing_rnn_size': [256],  # Decoder control preprocessing.
  })
  return contrib_training.HParams(**hparams_map)


class GrooveLstmDecoder(BaseLstmDecoder):
  """Groove LSTM decoder with MSE loss for continuous values.

  At each timestep, this decoder outputs a vector of length (N_INSTRUMENTS*3).
  The default number of drum instruments is 9, with drum categories defined in
  drums_encoder_decoder.py

  For each instrument, the model outputs a triple of (on/off, velocity, offset),
  with a binary representation for on/off, continuous values between 0 and 1
  for velocity, and continuous values between -0.5 and 0.5 for offset.
  """

  def _activate_outputs(self, flat_rnn_output):
    output_hits, output_velocities, output_offsets = tf_one.split(
        flat_rnn_output, 3, axis=1)

    output_hits = tf_one.nn.sigmoid(output_hits)
    output_velocities = tf_one.nn.sigmoid(output_velocities)
    output_offsets = tf_one.nn.tanh(output_offsets)

    return output_hits, output_velocities, output_offsets

  def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
    # flat_x_target is by default shape (1,27), [on/offs... vels...offsets...]
    # split into 3 equal length vectors
    target_hits, target_velocities, target_offsets = tf_one.split(
        flat_x_target, 3, axis=1)

    output_hits, output_velocities, output_offsets = self._activate_outputs(
        flat_rnn_output)

    hits_loss = tf_one.reduce_sum(tf_one.losses.log_loss(
        labels=target_hits, predictions=output_hits,
        reduction=tf_one.losses.Reduction.NONE), axis=1)

    velocities_loss = tf_one.reduce_sum(tf_one.losses.mean_squared_error(
        target_velocities, output_velocities,
        reduction=tf_one.losses.Reduction.NONE), axis=1)

    offsets_loss = tf_one.reduce_sum(tf_one.losses.mean_squared_error(
        target_offsets, output_offsets,
        reduction=tf_one.losses.Reduction.NONE), axis=1)

    loss = hits_loss + velocities_loss + offsets_loss

    metric_map = {
        'metrics/hits_loss':
            tf_one.metrics.mean(hits_loss),
        'metrics/velocities_loss':
            tf_one.metrics.mean(velocities_loss),
        'metrics/offsets_loss':
            tf_one.metrics.mean(offsets_loss)
    }

    return loss, metric_map

  def _sample(self, rnn_output, temperature=1.0):
    output_hits, output_velocities, output_offsets = tf_one.split(
        rnn_output, 3, axis=1)

    output_velocities = tf_one.nn.sigmoid(output_velocities)
    output_offsets = tf_one.nn.tanh(output_offsets)

    hits_sampler = tfp.distributions.Bernoulli(
        logits=output_hits / temperature, dtype=tf_one.float32)

    output_hits = hits_sampler.sample()
    return tf_one.concat([output_hits, output_velocities, output_offsets], axis=1)
