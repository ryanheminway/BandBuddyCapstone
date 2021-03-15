import tensorflow as tf
import configs as cfg

import NanoMagenta.nano_audio_utils as audio
import soundfile as sf
import trained_model_v2 as t2
import librosa
from base_model_v2 import GrooVAE

groovae_cfg = cfg.CONFIG_MAP['groovae_2bar_tap_fixed_velocity']
data_converter = groovae_cfg.data_converter
print("Loading a new model with trained weights")
latest_weights = tf.train.latest_checkpoint("./model_test_checkpoint/")
print("Latest checkpoint: ", latest_weights)
new_model = GrooVAE(groovae_cfg.hparams, data_converter.output_depth, False)
new_model.load_weights(latest_weights)

trained = t2.TrainedModel(groovae_cfg, new_model, 1, "./model_test_checkpoint/")
temperature = 0.8
velocity_threshold = 0.08

f = '../../Data/ryan_is_no_joe.wav'
y, sr = librosa.load(f)
print("Loaded wav data: ", y)
print("Length of wav data: ", len(y))
print("Loaded SR: ", sr)
# "TRANSLATE" DIRECTLY FROM AUDIO TO NEW DRUM TRACK
full_drum_audio = audio.audio_to_drum(y, sr, velocity_threshold=velocity_threshold,
                                      temperature=temperature,
                                      model=trained,
                                      v2=True)
print("what we got: ", full_drum_audio)

full_drum_audio = audio.midi_to_wav(full_drum_audio, sr)
sf.write("model_out_drums.wav", full_drum_audio, sr, subtype='PCM_24')
