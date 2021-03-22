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
  """
  def __init__(self, config, model, batch_size):
    self._config = copy.deepcopy(config)
    self._config.data_converter.set_mode('infer')
    self._config.hparams.batch_size = batch_size
    self.model = model

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
      if assert_same_length and len(inputs[0]) != len(inputs[-1]):
        raise AssertionError(
            'Sequences 0 and %d have different lengths: %d vs %d' %
            (len(inputs) - 1, len(inputs[0]), len(inputs[-1])))
    return self.encode_tensors(inputs)

  def encode_tensors(self, input_tensors):
    """Encodes a collection of input tensors into latent vectors.

    Args:
      input_tensors: Collection of input tensors to encode.
    Returns:
      The encoded `mu, `sigma, and `z` values.
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

    max_length = max([len(t) for t in input_tensors])
    inputs_array = np.zeros(
        [len(input_tensors), max_length, input_depth])
    for i, t in enumerate(input_tensors):
      inputs_array[i, :len(t)] = t

    self.inputs_array = inputs_array

    return self.model.encoder(inputs_array)

  def decode(self, z):
    """Decodes a collection of latent vectors into NoteSequences.

    Args:
      z: A collection of latent vectors to decode.
    Returns:
      A list of decodings as NoteSequence objects.
    Raises:
      RuntimeError: If called for a non-conditional model.
      ValueError: If `length` is not specified and an end token is not being
        used.
    """
    print("decode input array: ", tf.shape(self.inputs_array))
    tensors = self.decode_to_tensors(z, self.inputs_array)
    return self._config.data_converter.from_tensors(tensors.samples)

  def decode_to_tensors(self, z, x_input):
    """Decodes a collection of latent vectors into output tensors.

    Args:
      z: A collection of latent vectors to decode.
      x_input: the original input tensors passed to encoder
    Returns:
      Will return the samples from the decoder as a 2D numpy array.
    Raises:
      RuntimeError: If called for a non-conditional model.
    """
    if not self._config.hparams.z_size:
      raise RuntimeError('Cannot decode with a non-conditional model.')

    batch_size = self._config.hparams.batch_size
    n = len(z)

    batch_pad_amt = -n % batch_size
    z = np.pad(z, [(0, batch_pad_amt), (0, 0)], mode='constant')

    return self.model.decoder([z, x_input])

