import copy, librosa, numpy as np
import nano_configs as configs
from nano_trained_model import TrainedModel
import note_seq
from note_seq import midi_synth
from note_seq.protobuf import music_pb2
import soundfile as sf
import nano_audio_utils as audio

# Load model checkpoint
GROOVAE_2BAR_TAP_FIXED_VELOCITY = "../../model_checkpoints/groovae_rock/groovae_rock.tar"
config_2bar_tap = configs.CONFIG_MAP['groovae_2bar_tap_fixed_velocity']
# Build GrooVAE model from checkpoint variables and config model definition
groovae_2bar_tap = TrainedModel(config_2bar_tap, 1, checkpoint_dir_or_path=GROOVAE_2BAR_TAP_FIXED_VELOCITY)

paths = ['../../../Data/footsteps.wav']
temperature = 0.8
velocity_threshold = 0.08

new_beats = []
new_drum_audios = []
combined_audios = []

# Translate all input audio files to a new Drum track produced by TrainedModel
# Can assume, for now, that process was successful if ALL DONE is printed in console
for i in range(len(paths)):
    f = paths[i]
    y,sr = librosa.load(f)
    print("Loaded wav data: ", y)
    print("Length of wav data: ", len(y))
    print("Loaded SR: ", sr)
    # "TRANSLATE" DIRECTLY FROM AUDIO TO NEW DRUM TRACK
    full_drum_audio, _, tap_and_onsets, drums_and_original, _ = audio.audio_to_drum(y, sr,
                                                                                    velocity_threshold=velocity_threshold,
                                                                                    temperature=temperature,
                                                                                    model=groovae_2bar_tap)
    print("drum audio: ", type(full_drum_audio))
    print("combined audo", type(drums_and_original))
    # (TODO) lets me write the full_drum but not drums_and_original? Some diff in file type
    sf.write("model_out_drums.wav", full_drum_audio, sr, subtype='PCM_24')
    #sf.write("model_out_combined.wav", drums_and_original, sr, subtype='PCM_24')
print("ALL DONE")
