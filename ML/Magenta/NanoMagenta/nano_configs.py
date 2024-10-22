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
"""Configurations for MusicVAE models."""
import collections

from nano_magenta_hparams import HParams
import nano_data as data
import nano_lstm_models as lstm_models
from nano_base_model import MusicVAE


def merge_hparams(hparams_1, hparams_2):
  """Merge hyperparameters from two tf.contrib.training.HParams objects.

  If the same key is present in both HParams objects, the value from `hparams_2`
  will be used.

  Args:
    hparams_1: The first tf.contrib.training.HParams object to merge.
    hparams_2: The second tf.contrib.training.HParams object to merge.

  Returns:
    A merged tf.contrib.training.HParams object with the hyperparameters from
    both `hparams_1` and `hparams_2`.
  """
  hparams_map = hparams_1.values()
  hparams_map.update(hparams_2.values())
  return HParams(**hparams_map)


class Config(collections.namedtuple(
    'Config',
    ['model', 'hparams', 'note_sequence_augmenter', 'data_converter',
     'train_examples_path', 'eval_examples_path', 'tfds_name'])):

  def values(self):
    return self._asdict()


Config.__new__.__defaults__ = (None,) * len(Config._fields)


def update_config(config, update_dict):
  config_dict = config.values()
  config_dict.update(update_dict)
  return Config(**config_dict)


CONFIG_MAP = {}
CONFIG_MAP['groovae_2bar_tap_fixed_velocity'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.GrooveLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=1, # 512
            max_seq_len=16 * 2,  # 2 bars w/ 16 steps per bar
            z_size=256,
            enc_rnn_size=[512],
            dec_rnn_size=[256, 256],
            max_beta=0.2,
            free_bits=48,
            dropout_keep_prob=0.3,
        )),
    note_sequence_augmenter=None,
    data_converter=data.GrooveConverter(
        split_bars=2, steps_per_quarter=4, quarters_per_bar=4,
        max_tensors_per_notesequence=20, tapify=True, fixed_velocities=True,
        pitch_classes=data.ROLAND_DRUM_PITCH_CLASSES,
        inference_pitch_classes=data.REDUCED_DRUM_PITCH_CLASSES),
    tfds_name='groove/2bar-midionly'
)

