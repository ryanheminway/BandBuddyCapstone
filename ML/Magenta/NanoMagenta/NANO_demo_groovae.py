import copy, librosa, numpy as np
import NANO_configs as configs
from NANO_trained_model import TrainedModel
import note_seq
from note_seq import midi_synth
from note_seq.protobuf import music_pb2

# If a sequence has notes at time before 0.0, scoot them up to 0
def start_notes_at_0(s):
    for n in s.notes:
        if n.start_time < 0:
            n.end_time -= n.start_time
            n.start_time = 0
    return s

# quickly change the tempo of a midi sequence and adjust all notes
def change_tempo(note_sequence, new_tempo):
    new_sequence = copy.deepcopy(note_sequence)
    ratio = note_sequence.tempos[0].qpm / new_tempo
    for note in new_sequence.notes:
        note.start_time = note.start_time * ratio
        note.end_time = note.end_time * ratio
    new_sequence.tempos[0].qpm = new_tempo
    return new_sequence

def mix_tracks(y1, y2, stereo=False):
    l = max(len(y1), len(y2))
    y1 = librosa.util.fix_length(y1, l)
    y2 = librosa.util.fix_length(y2, l)

    if stereo:
        return np.vstack([y1, y2])
    else:
        return y1 + y2

def drumify(s, model, temperature=1.0):
    encoding, mu, sigma = model.encode([s])
    #print("ENCODED: ", len(encoding[0]))
    decoded = model.decode(encoding, length=32, temperature=temperature)
    #print("DECODED: ", decoded)
    return decoded[0]

def combine_sequences_with_lengths(sequences, lengths):
    seqs = copy.deepcopy(sequences)
    total_shift_amount = 0
    for i, seq in enumerate(seqs):
        if i == 0:
            shift_amount = 0
        else:
            shift_amount = lengths[i - 1]
        total_shift_amount += shift_amount
        if total_shift_amount > 0:
            seqs[i] = note_seq.sequences_lib.shift_sequence_times(seq, total_shift_amount)
    combined_seq = music_pb2.NoteSequence()
    for i in range(len(seqs)):
        tempo = combined_seq.tempos.add()
        tempo.qpm = seqs[i].tempos[0].qpm
        tempo.time = sum(lengths[0:i - 1])
        for note in seqs[i].notes:
            combined_seq.notes.extend([copy.deepcopy(note)])
    return combined_seq


# Allow encoding of a sequence that has no extracted examples
# by adding a quiet note after the desired length of time
def add_silent_note(note_sequence, num_bars):
    tempo = note_sequence.tempos[0].qpm
    length = 60 / tempo * 4 * num_bars
    note_sequence.notes.add(
        instrument=9, pitch=42, velocity=0, start_time=length - 0.02,
        end_time=length - 0.01, is_drum=True)


def get_bar_length(note_sequence):
    tempo = note_sequence.tempos[0].qpm
    return 60 / tempo * 4


def sequence_is_shorter_than_full(note_sequence):
    return note_sequence.notes[-1].start_time < get_bar_length(note_sequence)


def get_rhythm_elements(y, sr):
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, max_tempo=180)[0]
    onset_times = librosa.onset.onset_detect(y, sr, units='time')
    onset_frames = librosa.onset.onset_detect(y, sr, units='frames')
    onset_strengths = librosa.onset.onset_strength(y, sr)[onset_frames]
    normalized_onset_strengths = onset_strengths / np.max(onset_strengths)
    onset_velocities = np.int32(normalized_onset_strengths * 127)

    return tempo, onset_times, onset_frames, onset_velocities


def make_tap_sequence(tempo, onset_times, onset_frames, onset_velocities,
                      velocity_threshold, start_time, end_time):
    note_sequence = music_pb2.NoteSequence()
    note_sequence.tempos.add(qpm=tempo)
    for onset_vel, onset_time in zip(onset_velocities, onset_times):
        if onset_vel > velocity_threshold and onset_time >= start_time and onset_time < end_time:  # filter quietest notes
            note_sequence.notes.add(
                instrument=9, pitch=42, is_drum=True,
                velocity=onset_vel,  # model will use fixed velocity here
                start_time=onset_time - start_time,
                end_time=onset_time - start_time + 0.01
            )
    return note_sequence


# !!!!! TRANSLATE DIRECTLY FROM AUDIO TO DRUMS !!!!!
def audio_to_drum(f, velocity_threshold=30, temperature=1., force_sync=False, start_windows_on_downbeat=False):
    y, sr = librosa.load(f)
    # pad the beginning to avoid errors with onsets right at the start
    y = np.concatenate([np.zeros(1000), y])

    clip_length = float(len(y)) / sr

    tap_sequences = []
    # Loop through the file, grabbing 2-bar sections at a time, estimating
    # tempos along the way to try to handle tempo variations

    tempo, onset_times, onset_frames, onset_velocities = get_rhythm_elements(y, sr)

    initial_start_time = onset_times[0]

    start_time = onset_times[0]
    beat_length = 60 / tempo
    two_bar_length = beat_length * 8
    end_time = start_time + two_bar_length

    start_times = []
    lengths = []
    tempos = []

    start_times.append(start_time)
    lengths.append(end_time - start_time)
    tempos.append(tempo)

    tap_sequences.append(make_tap_sequence(tempo, onset_times, onset_frames,
                                           onset_velocities, velocity_threshold, start_time, end_time))

    start_time += two_bar_length;
    end_time += two_bar_length

    while start_time < clip_length:
        start_sample = int(librosa.core.time_to_samples(start_time, sr=sr))
        end_sample = int(librosa.core.time_to_samples(start_time + two_bar_length, sr=sr))
        current_section = y[start_sample:end_sample]
        tempo = librosa.beat.tempo(onset_envelope=librosa.onset.onset_strength(current_section, sr=sr), max_tempo=180)[
            0]

        beat_length = 60 / tempo
        two_bar_length = beat_length * 8

        end_time = start_time + two_bar_length

        start_times.append(start_time)
        lengths.append(end_time - start_time)
        tempos.append(tempo)

        tap_sequences.append(make_tap_sequence(tempo, onset_times, onset_frames,
                                               onset_velocities, velocity_threshold, start_time, end_time))

        start_time += two_bar_length;
        end_time += two_bar_length

    # if there's a long gap before the first note, back it up close to 0
    def _shift_notes_to_beginning(s):
        start_time = s.notes[0].start_time
        if start_time > 0.1:
            for n in s.notes:
                n.start_time -= start_time
                n.end_time -= start_time
        return start_time

    def _shift_notes_later(s, start_time):
        for n in s.notes:
            n.start_time += start_time
            n.end_time += start_time

    def _sync_notes_with_onsets(s, onset_times):
        for n in s.notes:
            n_length = n.end_time - n.start_time
            closest_onset_index = np.argmin(np.abs(n.start_time - onset_times))
            n.start_time = onset_times[closest_onset_index]
            n.end_time = n.start_time + n_length

    drum_seqs = []
    for s in tap_sequences:
        try:
            if sequence_is_shorter_than_full(s):
                add_silent_note(s, 2)

            if start_windows_on_downbeat:
                note_start_time = _shift_notes_to_beginning(s)

            h = drumify(s, groovae_2bar_tap, temperature=temperature)
            h = change_tempo(h, s.tempos[0].qpm)

            if start_windows_on_downbeat and note_start_time > 0.1:
                _shift_notes_later(s, note_start_time)

            drum_seqs.append(h)
        except:
            continue

    combined_tap_sequence = start_notes_at_0(combine_sequences_with_lengths(tap_sequences, lengths))
    combined_drum_sequence = start_notes_at_0(combine_sequences_with_lengths(drum_seqs, lengths))

    if force_sync:
        _sync_notes_with_onsets(combined_tap_sequence, onset_times)
        _sync_notes_with_onsets(combined_drum_sequence, onset_times)

    full_tap_audio = librosa.util.normalize(midi_synth.fluidsynth(combined_tap_sequence, sample_rate=sr))
    full_drum_audio = librosa.util.normalize(midi_synth.fluidsynth(combined_drum_sequence, sample_rate=sr))

    tap_and_onsets = mix_tracks(full_tap_audio, y[int(initial_start_time * sr):] / 2, stereo=True)
    drums_and_original = mix_tracks(full_drum_audio, y[int(initial_start_time * sr):] / 2, stereo=True)

    return full_drum_audio, full_tap_audio, tap_and_onsets, drums_and_original, combined_drum_sequence

"""
NOTE Ryan Heminway
Demonstrate interaction with GrooVAE (MusicVAE) model. Build model from saved checkpoint using Magenta's TrainedModel
interface. Using TrainedModel interface, we interact with the model to perform inference and produce new drum files from
an input of a wav file. 

See drumify() function for interaction with TrainedModel. Calls to Encode and Decode
"""

# Load model checkpoint
GROOVAE_2BAR_TAP_FIXED_VELOCITY = "../model_checkpoints/groovae_rock/groovae_rock.tar"
print("PATH: ", GROOVAE_2BAR_TAP_FIXED_VELOCITY)
config_2bar_tap = configs.CONFIG_MAP['groovae_2bar_tap_fixed_velocity']
# Build GrooVAE model from checkpoint variables and config model definition
groovae_2bar_tap = TrainedModel(config_2bar_tap, 1, checkpoint_dir_or_path=GROOVAE_2BAR_TAP_FIXED_VELOCITY)

paths = ['../../Data/basic_plain.wav']
temperature = 0.1
velocity_threshold = 0.08
stereo = False

new_beats = []
new_drum_audios = []
combined_audios = []

# Translate all input audio files to a new Drum track produced by TrainedModel
# Can assume, for now, that process was successful if ALL DONE is printed in console
#
for i in range(len(paths)):
    f = paths[i]
    y,sr = librosa.load(f)
    # "TRANSLATE" DIRECTLY FROM AUDIO TO NEW DRUM TRACK
    full_drum_audio, full_tap_audio, tap_and_onsets, drums_and_original, combined_drum_sequence = audio_to_drum(f, velocity_threshold=velocity_threshold, temperature=temperature)
    #print(full_tap_audio)
    #write("my_test.wav", sr, full_drum_audio)
    new_beats.append(combined_drum_sequence)
    new_drum_audios.append(full_drum_audio)
    combined_audios.append(drums_and_original)
    print("DONE")
print("ALL DONE")