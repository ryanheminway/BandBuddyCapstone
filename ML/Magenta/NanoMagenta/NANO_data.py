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

"""MusicVAE data library."""
import abc
import collections
import copy

import note_seq
from note_seq import drums_encoder_decoder
from note_seq import sequences_lib
import numpy as np
import tensorflow.compat.v1 as tf

PIANO_MIN_MIDI_PITCH = 21
PIANO_MAX_MIDI_PITCH = 108
MIN_MIDI_PITCH = 0
MAX_MIDI_PITCH = 127
MIDI_PITCHES = 128

MAX_INSTRUMENT_NUMBER = 127

MEL_PROGRAMS = range(0, 32)  # piano, chromatic percussion, organ, guitar
BASS_PROGRAMS = range(32, 40)
ELECTRIC_BASS_PROGRAM = 33

# 9 classes: kick, snare, closed_hh, open_hh, low_tom, mid_tom, hi_tom, crash,
# ride
REDUCED_DRUM_PITCH_CLASSES = drums_encoder_decoder.DEFAULT_DRUM_TYPE_PITCHES
# 61 classes: full General MIDI set
FULL_DRUM_PITCH_CLASSES = [
    [p] for p in  # pylint:disable=g-complex-comprehension
    [36, 35, 38, 27, 28, 31, 32, 33, 34, 37, 39, 40, 56, 65, 66, 75, 85, 42, 44,
     54, 68, 69, 70, 71, 73, 78, 80, 46, 67, 72, 74, 79, 81, 45, 29, 41, 61, 64,
     84, 48, 47, 60, 63, 77, 86, 87, 50, 30, 43, 62, 76, 83, 49, 55, 57, 58, 51,
     52, 53, 59, 82]
]
ROLAND_DRUM_PITCH_CLASSES = [
    # kick drum
    [36],
    # snare drum
    [38, 37, 40],
    # closed hi-hat
    [42, 22, 44],
    # open hi-hat
    [46, 26],
    # low tom
    [43, 58],
    # mid tom
    [47, 45],
    # high tom
    [50, 48],
    # crash cymbal
    [49, 52, 55, 57],
    # ride cymbal
    [51, 53, 59]
]

OUTPUT_VELOCITY = 80

CHORD_SYMBOL = note_seq.NoteSequence.TextAnnotation.CHORD_SYMBOL


def _maybe_pad_seqs(seqs, dtype, depth):
  """Pads sequences to match the longest and returns as a numpy array."""
  if not len(seqs):  # pylint:disable=g-explicit-length-test,len-as-condition
    return np.zeros((0, 0, depth), dtype)
  lengths = [len(s) for s in seqs]
  if len(set(lengths)) == 1:
    return np.array(seqs, dtype)
  else:
    length = max(lengths)
    return (np.array([np.pad(s, [(0, length - len(s)), (0, 0)], mode='constant')
                      for s in seqs], dtype))


def _extract_instrument(note_sequence, instrument):
  extracted_ns = copy.copy(note_sequence)
  del extracted_ns.notes[:]
  extracted_ns.notes.extend(
      n for n in note_sequence.notes if n.instrument == instrument)
  return extracted_ns


def maybe_sample_items(seq, sample_size, randomize):
  """Samples a seq if `sample_size` is provided and less than seq size."""
  if not sample_size or len(seq) <= sample_size:
    return seq
  if randomize:
    indices = set(np.random.choice(len(seq), size=sample_size, replace=False))
    return [seq[i] for i in indices]
  else:
    return seq[:sample_size]


def combine_converter_tensors(converter_tensors, max_num_tensors=None,
                              randomize_sample=True):
  """Combines multiple `ConverterTensors` into one and samples if required."""
  results = []
  for result in converter_tensors:
    results.extend(zip(*result))
  sampled_results = maybe_sample_items(results, max_num_tensors,
                                       randomize_sample)
  if sampled_results:
    return ConverterTensors(*zip(*sampled_results))
  else:
    return ConverterTensors()


def np_onehot(indices, depth, dtype=np.bool):
  """Converts 1D array of indices to a one-hot 2D array with given depth."""
  onehot_seq = np.zeros((len(indices), depth), dtype=dtype)
  onehot_seq[np.arange(len(indices)), indices] = 1.0
  return onehot_seq


class NoteSequenceAugmenter(object):
  """Class for augmenting NoteSequences.

  Attributes:
    transpose_range: A tuple containing the inclusive, integer range of
        transpose amounts to sample from. If None, no transposition is applied.
    stretch_range: A tuple containing the inclusive, float range of stretch
        amounts to sample from.
  Returns:
    The augmented NoteSequence.
  """

  def __init__(self, transpose_range=None, stretch_range=None):
    self._transpose_range = transpose_range
    self._stretch_range = stretch_range

  def augment(self, note_sequence):
    """Python implementation that augments the NoteSequence.

    Args:
      note_sequence: A NoteSequence proto to be augmented.

    Returns:
      The randomly augmented NoteSequence.
    """
    transpose_min, transpose_max = (
        self._transpose_range if self._transpose_range else (0, 0))
    stretch_min, stretch_max = (
        self._stretch_range if self._stretch_range else (1.0, 1.0))

    return sequences_lib.augment_note_sequence(
        note_sequence,
        stretch_min,
        stretch_max,
        transpose_min,
        transpose_max,
        delete_out_of_range_notes=True)

  def tf_augment(self, note_sequence_scalar):
    """TF op that augments the NoteSequence."""
    def _augment_str(note_sequence_str):
      note_sequence = note_seq.NoteSequence.FromString(
          note_sequence_str.numpy())
      augmented_ns = self.augment(note_sequence)
      return [augmented_ns.SerializeToString()]

    augmented_note_sequence_scalar = tf.py_function(
        _augment_str,
        inp=[note_sequence_scalar],
        Tout=tf.string,
        name='augment')
    augmented_note_sequence_scalar.set_shape(())
    return augmented_note_sequence_scalar


class ConverterTensors(collections.namedtuple(
    'ConverterTensors', ['inputs', 'outputs', 'controls', 'lengths'])):
  """Tuple of tensors output by `to_tensors` method in converters.

  Attributes:
    inputs: Input tensors to feed to the encoder.
    outputs: Output tensors to feed to the decoder.
    controls: (Optional) tensors to use as controls for both encoding and
        decoding.
    lengths: Length of each input/output/control sequence.
  """

  def __new__(cls, inputs=None, outputs=None, controls=None, lengths=None):
    if inputs is None:
      inputs = []
    if outputs is None:
      outputs = []
    if lengths is None:
      lengths = [len(i) for i in inputs]
    if not controls:
      controls = [np.zeros([l, 0]) for l in lengths]
    return super(ConverterTensors, cls).__new__(
        cls, inputs, outputs, controls, lengths)


class BaseNoteSequenceConverter(object):
  """Base class for data converters between items and tensors.

  Inheriting classes must implement the following abstract methods:
    -`to_tensors`
    -`from_tensors`
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self,
               input_depth,
               input_dtype,
               output_depth,
               output_dtype,
               control_depth=0,
               control_dtype=np.bool,
               end_token=None,
               max_tensors_per_notesequence=None,
               length_shape=(),
               presplit_on_time_changes=True):
    """Initializes BaseNoteSequenceConverter.

    Args:
      input_depth: Depth of final dimension of input (encoder) tensors.
      input_dtype: DType of input (encoder) tensors.
      output_depth: Depth of final dimension of output (decoder) tensors.
      output_dtype: DType of output (decoder) tensors.
      control_depth: Depth of final dimension of control tensors, or zero if not
          conditioning on control tensors.
      control_dtype: DType of control tensors.
      end_token: Optional end token.
      max_tensors_per_notesequence: The maximum number of outputs to return for
          each input.
      length_shape: Shape of length returned by `to_tensor`.
      presplit_on_time_changes: Whether to split NoteSequence on time changes
        before converting.
    """
    self._input_depth = input_depth
    self._input_dtype = input_dtype
    self._output_depth = output_depth
    self._output_dtype = output_dtype
    self._control_depth = control_depth
    self._control_dtype = control_dtype
    self._end_token = end_token
    self._max_tensors_per_input = max_tensors_per_notesequence
    self._str_to_item_fn = note_seq.NoteSequence.FromString
    self._mode = None
    self._length_shape = length_shape
    self._presplit_on_time_changes = presplit_on_time_changes

  def set_mode(self, mode):
    if mode not in ['train', 'eval', 'infer']:
      raise ValueError('Invalid mode: %s' % mode)
    self._mode = mode

  @property
  def is_training(self):
    return self._mode == 'train'

  @property
  def is_inferring(self):
    return self._mode == 'infer'

  @property
  def str_to_item_fn(self):
    return self._str_to_item_fn

  @property
  def max_tensors_per_notesequence(self):
    return self._max_tensors_per_input

  @max_tensors_per_notesequence.setter
  def max_tensors_per_notesequence(self, value):
    self._max_tensors_per_input = value

  @property
  def end_token(self):
    """End token, or None."""
    return self._end_token

  @property
  def input_depth(self):
    """Dimension of inputs (to encoder) at each timestep of the sequence."""
    return self._input_depth

  @property
  def input_dtype(self):
    """DType of inputs (to encoder)."""
    return self._input_dtype

  @property
  def output_depth(self):
    """Dimension of outputs (from decoder) at each timestep of the sequence."""
    return self._output_depth

  @property
  def output_dtype(self):
    """DType of outputs (from decoder)."""
    return self._output_dtype

  @property
  def control_depth(self):
    """Dimension of control inputs at each timestep of the sequence."""
    return self._control_depth

  @property
  def control_dtype(self):
    """DType of control inputs."""
    return self._control_dtype

  @property
  def length_shape(self):
    """Shape of length returned by `to_tensor`."""
    return self._length_shape

  @abc.abstractmethod
  def to_tensors(self, item):
    """Python method that converts `item` into list of `ConverterTensors`."""
    pass

  @abc.abstractmethod
  def from_tensors(self, samples, controls=None):
    """Python method that decodes model samples into list of items."""
    pass

class GrooveConverter(BaseNoteSequenceConverter):
  """Converts to and from hit/velocity/offset representations.

  In this setting, we represent drum sequences and performances
  as triples of (hit, velocity, offset). Each timestep refers to a fixed beat
  on a grid, which is by default spaced at 16th notes.  Drum hits that don't
  fall exactly on beat are represented through the offset value, which refers
  to the relative distance from the nearest quantized step.

  Hits are binary [0, 1].
  Velocities are continuous values in [0, 1].
  Offsets are continuous values in [-0.5, 0.5], rescaled to [-1, 1] for tensors.

  Each timestep contains this representation for each of a fixed list of
  drum categories, which by default is the list of 9 categories defined in
  drums_encoder_decoder.py.  With the default categories, the input and output
  at a single timestep is of length 9x3 = 27. So a single measure of drums
  at a 16th note grid is a matrix of shape (16, 27).

  Attributes:
    split_bars: Optional size of window to slide over full converted tensor.
    steps_per_quarter: The number of quantization steps per quarter note.
    quarters_per_bar: The number of quarter notes per bar.
    pitch_classes: A collection of collections, with each sub-collection
      containing the set of pitches representing a single class to group by. By
      default, groups Roland V-Drum pitches into 9 different classes.
    inference_pitch_classes: Pitch classes to use during inference. By default,
      uses same as `pitch_classes`.
    humanize: If True, flatten all input velocities and microtiming. The model
      then learns to map from a flattened input to the original sequence.
    tapify: If True, squash all drums at each timestep to the open hi-hat
      channel.
    add_instruments: A list of strings matching drums in DRUM_LIST.
      These drums are removed from the inputs but not the outputs.
    num_velocity_bins: The number of bins to use for representing velocity as
      one-hots.  If not defined, the converter will use continuous values.
    num_offset_bins: The number of bins to use for representing timing offsets
      as one-hots.  If not defined, the converter will use continuous values.
    split_instruments: Whether to produce outputs for each drum at a given
      timestep across multiple steps of the model output. With 9 drums, this
      makes the sequence 9 times as long. A one-hot control sequence is also
      created to identify which instrument is to be output at each step.
    hop_size: Number of steps to slide window.
    hits_as_controls: If True, pass in hits with the conditioning controls
      to force model to learn velocities and offsets.
    fixed_velocities: If True, flatten all input velocities.
    max_note_dropout_probability: If a value is provided, randomly drop out
      notes from the input sequences but not the output sequences.  On a per
      sequence basis, a dropout probability will be chosen uniformly between 0
      and this value such that some sequences will have fewer notes dropped
      out and some will have have more.  On a per note basis, lower velocity
      notes will be dropped out more often.
  """

  def __init__(self, split_bars=None, steps_per_quarter=4, quarters_per_bar=4,
               max_tensors_per_notesequence=8, pitch_classes=None,
               inference_pitch_classes=None, humanize=False, tapify=False,
               add_instruments=None, num_velocity_bins=None,
               num_offset_bins=None, split_instruments=False, hop_size=None,
               hits_as_controls=False, fixed_velocities=False,
               max_note_dropout_probability=None):

    self._split_bars = split_bars
    self._steps_per_quarter = steps_per_quarter
    self._steps_per_bar = steps_per_quarter * quarters_per_bar

    self._humanize = humanize
    self._tapify = tapify
    self._add_instruments = add_instruments
    self._fixed_velocities = fixed_velocities

    self._num_velocity_bins = num_velocity_bins
    self._num_offset_bins = num_offset_bins
    self._categorical_outputs = num_velocity_bins and num_offset_bins

    self._split_instruments = split_instruments

    self._hop_size = hop_size
    self._hits_as_controls = hits_as_controls

    def _classes_to_map(classes):
      class_map = {}
      for cls, pitches in enumerate(classes):
        for pitch in pitches:
          class_map[pitch] = cls
      return class_map

    self._pitch_classes = pitch_classes or ROLAND_DRUM_PITCH_CLASSES
    self._pitch_class_map = _classes_to_map(self._pitch_classes)
    self._infer_pitch_classes = inference_pitch_classes or self._pitch_classes
    self._infer_pitch_class_map = _classes_to_map(self._infer_pitch_classes)
    if len(self._pitch_classes) != len(self._infer_pitch_classes):
      raise ValueError(
          'Training and inference must have the same number of pitch classes. '
          'Got: %d vs %d.' % (
              len(self._pitch_classes), len(self._infer_pitch_classes)))
    self._num_drums = len(self._pitch_classes)

    if bool(num_velocity_bins) ^ bool(num_offset_bins):
      raise ValueError(
          'Cannot define only one of num_velocity_vins and num_offset_bins.')

    if split_bars is None and hop_size is not None:
      raise ValueError(
          'Cannot set hop_size without setting split_bars')

    drums_per_output = 1 if self._split_instruments else self._num_drums
    # Each drum hit is represented by 3 numbers - on/off, velocity, and offset
    if self._categorical_outputs:
      output_depth = (
          drums_per_output * (1 + num_velocity_bins + num_offset_bins))
    else:
      output_depth = drums_per_output * 3

    control_depth = 0
    # Set up controls for passing hits as side information.
    if self._hits_as_controls:
      if self._split_instruments:
        control_depth += 1
      else:
        control_depth += self._num_drums
    # Set up controls for cycling through instrument outputs.
    if self._split_instruments:
      control_depth += self._num_drums

    self._max_note_dropout_probability = max_note_dropout_probability
    self._note_dropout = max_note_dropout_probability is not None

    super(GrooveConverter, self).__init__(
        input_depth=output_depth,
        input_dtype=np.float32,
        output_depth=output_depth,
        output_dtype=np.float32,
        control_depth=control_depth,
        control_dtype=np.bool,
        end_token=False,
        presplit_on_time_changes=False,
        max_tensors_per_notesequence=max_tensors_per_notesequence)

  @property
  def pitch_classes(self):
    if self.is_inferring:
      return self._infer_pitch_classes
    return self._pitch_classes

  @property
  def pitch_class_map(self):  # pylint: disable=g-missing-from-attributes
    if self.is_inferring:
      return self._infer_pitch_class_map
    return self._pitch_class_map

  def _get_feature(self, note, feature, step_length=None):
    """Compute numeric value of hit/velocity/offset for a note.

    For now, only allow one note per instrument per quantization time step.
    This means at 16th note resolution we can't represent some drumrolls etc.
    We just take the note with the highest velocity if there are multiple notes.

    Args:
      note: A Note object from a NoteSequence.
      feature: A string, either 'hit', 'velocity', or 'offset'.
      step_length: Time duration in seconds of a quantized step. This only needs
        to be defined when the feature is 'offset'.

    Raises:
      ValueError: Any feature other than 'hit', 'velocity', or 'offset'.

    Returns:
      The numeric value of the feature for the note.
    """

    def _get_offset(note, step_length):
      true_onset = note.start_time
      quantized_onset = step_length * note.quantized_start_step
      diff = quantized_onset - true_onset
      return diff/step_length

    if feature == 'hit':
      if note:
        return 1.
      else:
        return 0.

    elif feature == 'velocity':
      if note:
        return note.velocity/127.  # Switch from [0, 127] to [0, 1] for tensors.
      else:
        return 0.  # Default velocity if there's no note is 0

    elif feature == 'offset':
      if note:
        offset = _get_offset(note, step_length)
        return offset*2  # Switch from [-0.5, 0.5] to [-1, 1] for tensors.
      else:
        return 0.  # Default offset if there's no note is 0

    else:
      raise ValueError('Unlisted feature: ' + feature)

  def to_tensors(self, note_sequence):

    def _get_steps_hash(note_sequence):
      """Partitions all Notes in a NoteSequence by quantization step and drum.

      Creates a hash with each hash bucket containing a dictionary
      of all the notes at one time step in the sequence grouped by drum/class.
      If there are no hits at a given time step, the hash value will be {}.

      Args:
        note_sequence: The NoteSequence object

      Returns:
        The fully constructed hash

      Raises:
        ValueError: If the sequence is not quantized
      """
      if not note_seq.sequences_lib.is_quantized_sequence(note_sequence):
        raise ValueError('NoteSequence must be quantized')

      h = collections.defaultdict(lambda: collections.defaultdict(list))

      for note in note_sequence.notes:
        step = int(note.quantized_start_step)
        drum = self.pitch_class_map[note.pitch]
        h[step][drum].append(note)

      return h

    def _remove_drums_from_tensors(to_remove, tensors):
      """Drop hits in drum_list and set velocities and offsets to 0."""
      for t in tensors:
        t[:, to_remove] = 0.
      return tensors

    def _convert_vector_to_categorical(vectors, min_value, max_value, num_bins):
      # Avoid edge case errors by adding a small amount to max_value
      bins = np.linspace(min_value, max_value+0.0001, num_bins)
      return np.array([np.concatenate(
          np_onehot(np.digitize(v, bins, right=True), num_bins, dtype=np.int32))
                       for v in vectors])

    def _extract_windows(tensor, window_size, hop_size):
      """Slide a window across the first dimension of a 2D tensor."""
      return [tensor[i:i+window_size, :] for i in range(
          0, len(tensor) - window_size  + 1, hop_size)]

    try:
      quantized_sequence = sequences_lib.quantize_note_sequence(
          note_sequence, self._steps_per_quarter)
      if (note_seq.steps_per_bar_in_quantized_sequence(quantized_sequence) !=
          self._steps_per_bar):
        return ConverterTensors()
      if not quantized_sequence.time_signatures:
        quantized_sequence.time_signatures.add(numerator=4, denominator=4)
    except (note_seq.BadTimeSignatureError, note_seq.NonIntegerStepsPerBarError,
            note_seq.NegativeTimeError, note_seq.MultipleTimeSignatureError,
            note_seq.MultipleTempoError):
      return ConverterTensors()

    beat_length = 60. / quantized_sequence.tempos[0].qpm
    step_length = beat_length / (
        quantized_sequence.quantization_info.steps_per_quarter)

    steps_hash = _get_steps_hash(quantized_sequence)

    if not quantized_sequence.notes:
      return ConverterTensors()

    max_start_step = np.max(
        [note.quantized_start_step for note in quantized_sequence.notes])

    # Round up so we pad to the end of the bar.
    total_bars = int(np.ceil((max_start_step + 1) / self._steps_per_bar))
    max_step = self._steps_per_bar * total_bars

    # Each of these stores a (total_beats, num_drums) matrix.
    hit_vectors = np.zeros((max_step, self._num_drums))
    velocity_vectors = np.zeros((max_step, self._num_drums))
    offset_vectors = np.zeros((max_step, self._num_drums))

    # Loop through timesteps.
    for step in range(max_step):
      notes = steps_hash[step]

      # Loop through each drum instrument.
      for drum in range(self._num_drums):
        drum_notes = notes[drum]
        if len(drum_notes) > 1:
          note = max(drum_notes, key=lambda n: n.velocity)
        elif len(drum_notes) == 1:
          note = drum_notes[0]
        else:
          note = None

        hit_vectors[step, drum] = self._get_feature(note, 'hit')
        velocity_vectors[step, drum] = self._get_feature(note, 'velocity')
        offset_vectors[step, drum] = self._get_feature(
            note, 'offset', step_length)

    # These are the input tensors for the encoder.
    in_hits = copy.deepcopy(hit_vectors)
    in_velocities = copy.deepcopy(velocity_vectors)
    in_offsets = copy.deepcopy(offset_vectors)

    if self._note_dropout:
      # Choose a uniform dropout probability for notes per sequence.
      note_dropout_probability = np.random.uniform(
          0.0, self._max_note_dropout_probability)
      # Drop out lower velocity notes with higher probability.
      velocity_dropout_weights = np.maximum(0.2, (1 - in_velocities))
      note_dropout_keep_mask = 1 - np.random.binomial(
          1, velocity_dropout_weights * note_dropout_probability)
      in_hits *= note_dropout_keep_mask
      in_velocities *= note_dropout_keep_mask
      in_offsets *= note_dropout_keep_mask

    if self._tapify:
      argmaxes = np.argmax(in_velocities, axis=1)
      in_hits[:] = 0
      in_velocities[:] = 0
      in_offsets[:] = 0
      in_hits[:, 3] = hit_vectors[np.arange(max_step), argmaxes]
      in_velocities[:, 3] = velocity_vectors[np.arange(max_step), argmaxes]
      in_offsets[:, 3] = offset_vectors[np.arange(max_step), argmaxes]

    if self._humanize:
      in_velocities[:] = 0
      in_offsets[:] = 0

    if self._fixed_velocities:
      in_velocities[:] = 0

    # If learning to add drums, remove the specified drums from the inputs.
    if self._add_instruments:
      in_hits, in_velocities, in_offsets = _remove_drums_from_tensors(
          self._add_instruments, [in_hits, in_velocities, in_offsets])

    if self._categorical_outputs:
      # Convert continuous velocity and offset to one hots.
      velocity_vectors = _convert_vector_to_categorical(
          velocity_vectors, 0., 1., self._num_velocity_bins)
      in_velocities = _convert_vector_to_categorical(
          in_velocities, 0., 1., self._num_velocity_bins)

      offset_vectors = _convert_vector_to_categorical(
          offset_vectors, -1., 1., self._num_offset_bins)
      in_offsets = _convert_vector_to_categorical(
          in_offsets, -1., 1., self._num_offset_bins)

    if self._split_instruments:
      # Split the outputs for each drum into separate steps.
      total_length = max_step * self._num_drums
      hit_vectors = hit_vectors.reshape([total_length, -1])
      velocity_vectors = velocity_vectors.reshape([total_length, -1])
      offset_vectors = offset_vectors.reshape([total_length, -1])
      in_hits = in_hits.reshape([total_length, -1])
      in_velocities = in_velocities.reshape([total_length, -1])
      in_offsets = in_offsets.reshape([total_length, -1])
    else:
      total_length = max_step

    # Now concatenate all 3 vectors into 1, eg (16, 27).
    seqs = np.concatenate(
        [hit_vectors, velocity_vectors, offset_vectors], axis=1)

    input_seqs = np.concatenate(
        [in_hits, in_velocities, in_offsets], axis=1)

    # Controls section.
    controls = []
    if self._hits_as_controls:
      controls.append(hit_vectors.astype(np.bool))
    if self._split_instruments:
      # Cycle through instrument numbers.
      controls.append(np.tile(
          np_onehot(
              np.arange(self._num_drums), self._num_drums, np.bool),
          (max_step, 1)))
    controls = np.concatenate(controls, axis=-1) if controls else None

    if self._split_bars:
      window_size = self._steps_per_bar * self._split_bars
      hop_size = self._hop_size or window_size
      if self._split_instruments:
        window_size *= self._num_drums
        hop_size *= self._num_drums
      seqs = _extract_windows(seqs, window_size, hop_size)
      input_seqs = _extract_windows(input_seqs, window_size, hop_size)
      if controls is not None:
        controls = _extract_windows(controls, window_size, hop_size)
    else:
      # Output shape will look like (1, 64, output_depth).
      seqs = [seqs]
      input_seqs = [input_seqs]
      if controls is not None:
        controls = [controls]

    return ConverterTensors(inputs=input_seqs, outputs=seqs, controls=controls)

  def from_tensors(self, samples, controls=None):

    def _zero_one_to_velocity(val):
      output = int(np.round(val*127))
      return np.clip(output, 0, 127)

    def _minus_1_1_to_offset(val):
      output = val/2
      return np.clip(output, -0.5, 0.5)

    def _one_hot_to_velocity(v):
      return int((np.argmax(v) / len(v)) * 127)

    def _one_hot_to_offset(v):
      return (np.argmax(v) / len(v)) - 0.5

    output_sequences = []

    for sample in samples:
      n_timesteps = (sample.shape[0] // (
          self._num_drums if self._categorical_outputs else 1))

      note_sequence = note_seq.NoteSequence()
      note_sequence.tempos.add(qpm=120)
      beat_length = 60. / note_sequence.tempos[0].qpm
      step_length = beat_length / self._steps_per_quarter

      # Each timestep should be a (1, output_depth) vector
      # representing n hits, n velocities, and n offsets in order.

      for i in range(n_timesteps):
        if self._categorical_outputs:
          # Split out the categories from the flat output.
          if self._split_instruments:
            hits, velocities, offsets = np.split(  # pylint: disable=unbalanced-tuple-unpacking
                sample[i*self._num_drums: (i+1)*self._num_drums],
                [1, self._num_velocity_bins+1],
                axis=1)
          else:
            hits, velocities, offsets = np.split(  # pylint: disable=unbalanced-tuple-unpacking
                sample[i],
                [self._num_drums, self._num_drums*(self._num_velocity_bins+1)]
                )
            # Split out the instruments.
            velocities = np.split(velocities, self._num_drums)
            offsets = np.split(offsets, self._num_drums)
        else:
          if self._split_instruments:
            hits, velocities, offsets = sample[
                i*self._num_drums: (i+1)*self._num_drums].T
          else:
            hits, velocities, offsets = np.split(  # pylint: disable=unbalanced-tuple-unpacking
                sample[i], 3)

        # Loop through the drum instruments: kick, snare, etc.
        for j in range(len(hits)):
          # Create a new note
          if hits[j] > 0.5:
            note = note_sequence.notes.add()
            note.instrument = 9  # All drums are instrument 9
            note.is_drum = True
            pitch = self.pitch_classes[j][0]
            note.pitch = pitch
            if self._categorical_outputs:
              note.velocity = _one_hot_to_velocity(velocities[j])
              offset = _one_hot_to_offset(offsets[j])
            else:
              note.velocity = _zero_one_to_velocity(velocities[j])
              offset = _minus_1_1_to_offset(offsets[j])
            note.start_time = (i - offset) * step_length
            note.end_time = note.start_time + step_length

      output_sequences.append(note_sequence)

    return output_sequences


def split_process_and_combine(note_sequence, split, sample_size, randomize,
                              to_tensors_fn):
  """Splits a `NoteSequence`, processes and combines the `ConverterTensors`.

  Args:
    note_sequence: The `NoteSequence` to split, process and combine.
    split: If True, the given note_sequence is split into multiple based on time
      changes, and the tensor outputs are concatenated.
    sample_size: Outputs are sampled if size exceeds this value.
    randomize: If True, outputs are randomly sampled (this is generally done
      during training).
    to_tensors_fn: A fn that converts a `NoteSequence` to `ConverterTensors`.

  Returns:
    A `ConverterTensors` obj.
  """
  note_sequences = sequences_lib.split_note_sequence_on_time_changes(
      note_sequence) if split else [note_sequence]
  results = []
  for ns in note_sequences:
    tensors = to_tensors_fn(ns)
    sampled_results = maybe_sample_items(
        list(zip(*tensors)), sample_size, randomize)
    if sampled_results:
      results.append(ConverterTensors(*zip(*sampled_results)))
    else:
      results.append(ConverterTensors())
  return combine_converter_tensors(results, sample_size, randomize)


def convert_to_tensors_op(item_scalar, converter):
  """TensorFlow op that converts item into output tensors.

  Sequences will be padded to match the length of the longest.

  Args:
    item_scalar: A scalar of type tf.String containing the raw item to be
      converted to tensors.
    converter: The DataConverter to be used.

  Returns:
    inputs: A Tensor, shaped [num encoded seqs, max(lengths), input_depth],
        containing the padded input encodings.
    outputs: A Tensor, shaped [num encoded seqs, max(lengths), output_depth],
        containing the padded output encodings resulting from the input.
    controls: A Tensor, shaped
        [num encoded seqs, max(lengths), control_depth], containing the padded
        control encodings.
    lengths: A tf.int32 Tensor, shaped [num encoded seqs], containing the
      unpadded lengths of the tensor sequences resulting from the input.
  """

  def _convert_and_pad(item_str):
    item = converter.str_to_item_fn(item_str.numpy())  # pylint:disable=not-callable
    tensors = converter.to_tensors(item)
    inputs = _maybe_pad_seqs(tensors.inputs, converter.input_dtype,
                             converter.input_depth)
    outputs = _maybe_pad_seqs(tensors.outputs, converter.output_dtype,
                              converter.output_depth)
    controls = _maybe_pad_seqs(tensors.controls, converter.control_dtype,
                               converter.control_depth)
    return inputs, outputs, controls, np.array(tensors.lengths, np.int32)

  inputs, outputs, controls, lengths = tf.py_function(
      _convert_and_pad,
      inp=[item_scalar],
      Tout=[
          converter.input_dtype, converter.output_dtype,
          converter.control_dtype, tf.int32
      ],
      name='convert_and_pad')
  inputs.set_shape([None, None, converter.input_depth])
  outputs.set_shape([None, None, converter.output_depth])
  controls.set_shape([None, None, converter.control_depth])
  lengths.set_shape([None] + list(converter.length_shape))
  return inputs, outputs, controls, lengths
