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
"""A class for sampling, encoding, and decoding from trained MusicVAE models.

Adjusted specifically to run on the Jetson Nano. Removed all mention of Magenta library.
"""
import copy
import os
import tarfile
import tempfile
import re

import numpy as np
import tensorflow as tf


class NoExtractedExamplesError(Exception):
  pass


class MultipleExtractedExamplesError(Exception):
  pass


class TrainedModel(object):
  """An interface to a trained model for encoding, decoding, and sampling.

  Attributes:
    config: The Config to build the model graph with.
    batch_size: The batch size to build the model graph with.
    checkpoint_dir_or_path: The directory containing checkpoints for the model,
      the most recent of which will be loaded, or a direct path to a specific
      checkpoint.
  """

  def __init__(self, config, model, batch_size, checkpoint_dir_path):
    self._config = copy.deepcopy(config)
    self._config.data_converter.set_mode('infer')
    self._config.hparams.batch_size = batch_size
    self.model = model
    #latest = tf.train.latest_checkpoint(checkpoint_dir_path)
    #self.model.load_weights(latest)

  def encode(self, note_sequences, assert_same_length=False):
    """Encodes a collection of NoteSequences into latent vectors.

    Args:
      note_sequences: A collection of NoteSequence objects to encode.
      assert_same_length: Whether to raise an AssertionError if all of the
        extracted sequences are not the same length.
    Returns:
      The encoded `z`, `mu`, and `sigma` values.
    Raises:
      RuntimeError: If called for a non-conditional model.
      NoExtractedExamplesError: If no examples were extracted.
      MultipleExtractedExamplesError: If multiple examples were extracted.
      AssertionError: If `assert_same_length` is True and any extracted
        sequences differ in length.
    """
    if not self._config.hparams.z_size:
      raise RuntimeError('Cannot encode with a non-conditional model.')

    inputs = []
    controls = []
    lengths = []
    for note_sequence in note_sequences:
      extracted_tensors = self._config.data_converter.to_tensors(note_sequence)
      if not extracted_tensors.inputs:
        raise NoExtractedExamplesError(
            'No examples extracted from NoteSequence: %s' % note_sequence)
      if len(extracted_tensors.inputs) > 1:
        raise MultipleExtractedExamplesError(
            'Multiple (%d) examples extracted from NoteSequence: %s' %
            (len(extracted_tensors.inputs), note_sequence))
      inputs.append(extracted_tensors.inputs[0])
      controls.append(extracted_tensors.controls[0])
      lengths.append(extracted_tensors.lengths[0])
      if assert_same_length and len(inputs[0]) != len(inputs[-1]):
        raise AssertionError(
            'Sequences 0 and %d have different lengths: %d vs %d' %
            (len(inputs) - 1, len(inputs[0]), len(inputs[-1])))
    return self.encode_tensors(inputs, lengths, controls)

  def encode_tensors(self, input_tensors, lengths, control_tensors=None):
    """Encodes a collection of input tensors into latent vectors.

    Args:
      input_tensors: Collection of input tensors to encode.
      lengths: Collection of lengths of input tensors.
      control_tensors: Collection of control tensors to encode.
    Returns:
      The encoded `z`, `mu`, and `sigma` values.
    Raises:
       RuntimeError: If called for a non-conditional model.
    """
    if not self._config.hparams.z_size:
      raise RuntimeError('Cannot encode with a non-conditional model.')

    n = len(input_tensors)
    input_depth = self._config.data_converter.input_depth
    batch_size = self._config.hparams.batch_size

    batch_pad_amt = -n % batch_size
    if batch_pad_amt > 0:
      input_tensors += [np.zeros([0, input_depth])] * batch_pad_amt
    length_array = np.array(lengths, np.int32)
    length_array = np.pad(
        length_array,
        [(0, batch_pad_amt)] + [(0, 0)] * (length_array.ndim - 1),
        'constant')

    max_length = max([len(t) for t in input_tensors])
    inputs_array = np.zeros(
        [len(input_tensors), max_length, input_depth])
    for i, t in enumerate(input_tensors):
      inputs_array[i, :len(t)] = t

    control_depth = self._config.data_converter.control_depth
    controls_array = np.zeros(
        [len(input_tensors), max_length, control_depth])
    if control_tensors is not None:
      control_tensors += [np.zeros([0, control_depth])] * batch_pad_amt
      for i, t in enumerate(control_tensors):
        controls_array[i, :len(t)] = t

    return self.model.encode(inputs_array)

  def decode(self, z, x_input, length=None, temperature=1.0, c_input=None):
    """Decodes a collection of latent vectors into NoteSequences.

    Args:
      z: A collection of latent vectors to decode.
      length: The maximum length of a sample in decoder iterations. Required
        if end tokens are not being used.
      temperature: The softmax temperature to use (if applicable).
      c_input: Control sequence (if applicable).
    Returns:
      A list of decodings as NoteSequence objects.
    Raises:
      RuntimeError: If called for a non-conditional model.
      ValueError: If `length` is not specified and an end token is not being
        used.
    """
    tensors = self.decode_to_tensors(z, x_input, length, temperature, c_input)
    return self._config.data_converter.from_tensors(tensors.samples)

  def decode_to_tensors(self, z, x_input, length=None, temperature=1.0, c_input=None,
                        return_full_results=False):
    """Decodes a collection of latent vectors into output tensors.

    Args:
      z: A collection of latent vectors to decode.
      length: The maximum length of a sample in decoder iterations. Required
        if end tokens are not being used.
      temperature: The softmax temperature to use (if applicable).
      c_input: Control sequence (if applicable).
      return_full_results: If true will return the full decoder_results,
        otherwise it will return only the samples.
    Returns:
      If return_full_results is True, will return the full decoder_results list,
      otherwise it will return the samples from the decoder as a 2D numpy array.
    Raises:
      RuntimeError: If called for a non-conditional model.
      ValueError: If `length` is not specified and an end token is not being
        used.
    """
    if not self._config.hparams.z_size:
      raise RuntimeError('Cannot decode with a non-conditional model.')

    if not length and self._config.data_converter.end_token is None:
      raise ValueError(
          'A length must be specified when the end token is not used.')
    batch_size = self._config.hparams.batch_size
    n = len(z)
    length = length or tf.int32.max

    batch_pad_amt = -n % batch_size
    z = np.pad(z, [(0, batch_pad_amt), (0, 0)], mode='constant')

    return self.model.decoder([z, x_input])

