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

import NANO_rnn as nano_rnn
import NANO_seq2seq as contrib_seq2seq
import NANO_base_model as base_model
import NANO_lstm_utils as lstm_utils
import NANO_configs as configs
from NANO_common import flatten_maybe_padded_sequences
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


# (NOTE Ryan Heminway)
# Tried to remove everything that is unused by our GrooVAE model
class BidirectionalLstmEncoder(base_model.BaseEncoder):
  """Bidirectional LSTM Encoder."""

  @property
  def output_depth(self):
    return self._cells[0][-1].output_size + self._cells[1][-1].output_size

  def build(self, hparams, is_training=True, name_or_scope='encoder'):
    self._is_training = is_training
    self._name_or_scope = name_or_scope
    if hparams.use_cudnn:
      tf.logging.warning('cuDNN LSTM no longer supported. Using regular LSTM.')

    tf.logging.info('\nEncoder Cells (bidirectional):\n'
                    '  units: %s\n',
                    hparams.enc_rnn_size)

    self._cells = lstm_utils.build_bidirectional_lstm(
        layer_sizes=hparams.enc_rnn_size,
        dropout_keep_prob=hparams.dropout_keep_prob,
        residual=hparams.residual_encoder,
        is_training=is_training)

  def encode(self, sequence, sequence_length):
    cells_fw, cells_bw = self._cells

    output, states_fw, states_bw = nano_rnn.stack_bidirectional_dynamic_rnn(
        cells_fw,
        cells_bw,
        sequence,
        sequence_length=sequence_length,
        time_major=False,
        dtype=tf.float32,
        scope=self._name_or_scope)
    # Note we access the outputs (h) from the states since the backward
    # outputs are reversed to the input order in the returned outputs.

    # (NOTE Ryan Heminway) changed syntax for accessing outputs based on change to LSTM cell definition
    #   Keras LSTMCell states do not have a .h property
    #last_h_fw = states_fw[-1][-1].h
    #last_h_bw = states_bw[-1][-1].h
    last_h_fw = states_fw[-1][-1][-1]
    last_h_bw = states_bw[-1][-1][-1]

    return tf.concat([last_h_fw, last_h_bw], 1)


# DECODERS

class BaseLstmDecoder(base_model.BaseDecoder):
  """Abstract LSTM Decoder class.

  Implementations must define the following abstract methods:
      -`_sample`
      -`_flat_reconstruction_loss`
  """

  def build(self, hparams, output_depth, is_training=True):
    if hparams.use_cudnn:
      tf.logging.warning('cuDNN LSTM no longer supported. Using regular LSTM.')

    self._is_training = is_training

    tf.logging.info('\nDecoder Cells:\n'
                    '  units: %s\n',
                    hparams.dec_rnn_size)

    self._sampling_probability = lstm_utils.get_sampling_probability(
        hparams, is_training)
    self._output_depth = output_depth
    self._output_layer = tf.layers.Dense(
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
    z = tf.zeros([batch_size, 0]) if z is None else z
    repeated_z = tf.tile(
        tf.expand_dims(z, axis=1), [1, tf.shape(x_input)[1], 1])

    has_control = c_input is not None
    if c_input is None:
      c_input = tf.zeros([batch_size, tf.shape(x_input)[1], 0])

    sampling_probability_static = tf.get_static_value(
        self._sampling_probability)
    if sampling_probability_static == 0.0:
      # Use teacher forcing.
      x_input = tf.concat([x_input, repeated_z, c_input], axis=2)
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
    cum_x_len = tf.concat([(0,), tf.cumsum(x_length)], axis=0)
    r_losses = []
    for i in range(batch_size):
      b, e = cum_x_len[i], cum_x_len[i + 1]
      r_losses.append(tf.reduce_sum(r_loss[b:e]))
    r_loss = tf.stack(r_losses)

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
    z = tf.zeros((n, 0), tf.float32) if z is None else z

    if c_input is not None:
      # Tile control sequence across samples.
      c_input = tf.tile(tf.expand_dims(c_input, 1), [1, n, 1])

    # If not given, start with zeros.
    if start_inputs is None:
      start_inputs = tf.zeros([n, self._output_depth], dtype=tf.float32)
    # In the conditional case, also concatenate the Z.
    start_inputs = tf.concat([start_inputs, z], axis=-1)
    if c_input is not None:
      start_inputs = tf.concat([start_inputs, c_input[0]], axis=-1)
    initialize_fn = lambda: (tf.zeros([n], tf.bool), start_inputs)

    sample_fn = lambda time, outputs, state: self._sample(outputs, temperature)
    end_fn = end_fn or (lambda x: False)

    def next_inputs_fn(time, outputs, state, sample_ids):
      del outputs
      finished = end_fn(sample_ids)
      next_inputs = tf.concat([sample_ids, z], axis=-1)
      if c_input is not None:
        # We need to stop if we've run out of control input.
        finished = tf.cond(tf.less(time, tf.shape(c_input)[0] - 1),
                           lambda: finished,
                           lambda: True)
        next_inputs = tf.concat([
            next_inputs,
            tf.cond(tf.less(time, tf.shape(c_input)[0] - 1),
                    lambda: c_input[time + 1],
                    lambda: tf.zeros_like(c_input[0]))  # should be unused
        ], axis=-1)
      return (finished, next_inputs, state)

    sampler = contrib_seq2seq.CustomHelper(
        initialize_fn=initialize_fn, sample_fn=sample_fn,
        next_inputs_fn=next_inputs_fn, sample_ids_shape=[self._output_depth],
        sample_ids_dtype=tf.float32)

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
  return configs.HParams(**hparams_map)


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
    output_hits, output_velocities, output_offsets = tf.split(
        flat_rnn_output, 3, axis=1)

    output_hits = tf.nn.sigmoid(output_hits)
    output_velocities = tf.nn.sigmoid(output_velocities)
    output_offsets = tf.nn.tanh(output_offsets)

    return output_hits, output_velocities, output_offsets

  def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
    # flat_x_target is by default shape (1,27), [on/offs... vels...offsets...]
    # split into 3 equal length vectors
    target_hits, target_velocities, target_offsets = tf.split(
        flat_x_target, 3, axis=1)

    output_hits, output_velocities, output_offsets = self._activate_outputs(
        flat_rnn_output)

    hits_loss = tf.reduce_sum(tf.losses.log_loss(
        labels=target_hits, predictions=output_hits,
        reduction=tf.losses.Reduction.NONE), axis=1)

    velocities_loss = tf.reduce_sum(tf.losses.mean_squared_error(
        target_velocities, output_velocities,
        reduction=tf.losses.Reduction.NONE), axis=1)

    offsets_loss = tf.reduce_sum(tf.losses.mean_squared_error(
        target_offsets, output_offsets,
        reduction=tf.losses.Reduction.NONE), axis=1)

    loss = hits_loss + velocities_loss + offsets_loss

    metric_map = {
        'metrics/hits_loss':
            tf.metrics.mean(hits_loss),
        'metrics/velocities_loss':
            tf.metrics.mean(velocities_loss),
        'metrics/offsets_loss':
            tf.metrics.mean(offsets_loss)
    }

    return loss, metric_map

  def _sample(self, rnn_output, temperature=1.0):
    output_hits, output_velocities, output_offsets = tf.split(
        rnn_output, 3, axis=1)

    output_velocities = tf.nn.sigmoid(output_velocities)
    output_offsets = tf.nn.tanh(output_offsets)

    hits_sampler = tfp.distributions.Bernoulli(
        logits=output_hits / temperature, dtype=tf.float32)

    output_hits = hits_sampler.sample()
    return tf.concat([output_hits, output_velocities, output_offsets], axis=1)