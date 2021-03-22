import copy, librosa, numpy as np
import note_seq
from note_seq import midi_synth
from note_seq.protobuf import music_pb2
import soundfile as sf
import io

# Read WAV data from network message into NP array
# Based on librosa implementation (https://github.com/librosa)
def read_wav_data(data, mono=True, dtype=np.float32):
    np_wav_data, sr = sf.read(io.BytesIO(data))
    # Transpose to match librosa style, which our models expect
    np_wav_data = np_wav_data.T
    
    # Convert from stereo to mono track, if applicable
    if mono:
        # Ensure Fortran contiguity
        np_wav_data = np.asfortranarray(np_wav_data)
    if np_wav_data.ndim > 1:
        np_wav_data = np.mean(np_wav_data, axis=0)

    return np_wav_data, sr


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


# Combines two tracks in to one
def mix_tracks(y1, y2, stereo=False):
    l = max(len(y1), len(y2))
    y1 = librosa.util.fix_length(y1, l)
    y2 = librosa.util.fix_length(y2, l)

    if stereo:
        return np.vstack([y1, y2])
    else:
        return y1 + y2


# Apply GrooVAE model to input tapped sequence
def drumify(s, model, temperature=1.0):
    encoding, mu, sigma = model.encode([s])
    decoded = model.decode(encoding, length=32, temperature=temperature)
    return decoded[0]


def drumify_v2(s, model, temperature=1.0):
    mu, sigma, encoding = model.encode([s])
    decoded = model.decode(encoding)
    return decoded[0]


# Combine an ordered list of sequences into one sequence
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


# Gets bar length in seconds based on input tempo
def get_bar_length(note_sequence):
    tempo = note_sequence.tempos[0].qpm
    return 60 / tempo * 4


# Does the sequence end earlier than a full bar?
def sequence_is_shorter_than_full(note_sequence):
    return note_sequence.notes[-1].start_time < get_bar_length(note_sequence)


# Get onset times, frames, and velocities from a wav file. Given a sampling rate.
def get_rhythm_elements(y, sr):
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, max_tempo=180)[0]
    onset_times = librosa.onset.onset_detect(y, sr, units='time')
    onset_frames = librosa.onset.onset_detect(y, sr, units='frames')
    onset_strengths = librosa.onset.onset_strength(y, sr)[onset_frames]
    normalized_onset_strengths = onset_strengths / np.max(onset_strengths)
    onset_velocities = np.int32(normalized_onset_strengths * 127)

    return tempo, onset_times, onset_frames, onset_velocities


# Given characteristics of a drum beat (onset times, frames, velocities, tempo), returns a NoteSequence
# pattern which represents a "tapped" version of the input beat. "Tapped" meaning that each pulse is identical pitch.
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


# Given a .wav file path, applies the Drumify model to the input track and outputs a drum track.
def audio_to_drum(y, sr, velocity_threshold, temperature, model, v2=False, force_sync=False, start_windows_on_downbeat=False):
    #y, sr = librosa.load(f)
    
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
    print("tap sequences: ", tap_sequences)
    while start_time < clip_length:
        print("were here right?")
        start_sample = int(librosa.core.time_to_samples(start_time, sr=sr))
        end_sample = int(librosa.core.time_to_samples(start_time + two_bar_length, sr=sr))
        current_section = y[start_sample:end_sample]
        # Approximate tempo
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

            if v2:
                h = drumify_v2(s, model, temperature)
            else:
                h = drumify(s, model, temperature=temperature)
            print("Got drumify result: ", h)
            # Adjust drum output to input tempo
            h = change_tempo(h, s.tempos[0].qpm)

            if start_windows_on_downbeat and note_start_time > 0.1:
                _shift_notes_later(s, note_start_time)

            drum_seqs.append(h)
        except Exception as e:
            raise e
            print("got exception: ", e)
            continue

    combined_tap_sequence = start_notes_at_0(combine_sequences_with_lengths(tap_sequences, lengths))
    combined_drum_sequence = start_notes_at_0(combine_sequences_with_lengths(drum_seqs, lengths))

    if force_sync:
        _sync_notes_with_onsets(combined_tap_sequence, onset_times)
        _sync_notes_with_onsets(combined_drum_sequence, onset_times)

    return combined_drum_sequence

    """ Disabling for now, may not need at all
    # (TODO) Add additional SF2 soundpacks for variation so its not always default
    full_tap_audio = librosa.util.normalize(midi_synth.fluidsynth(combined_tap_sequence, sample_rate=sr))
    full_drum_audio = librosa.util.normalize(midi_synth.fluidsynth(combined_drum_sequence, sample_rate=sr))

    tap_and_onsets = mix_tracks(full_tap_audio, y[int(initial_start_time * sr):] / 2, stereo=True)
    drums_and_original = mix_tracks(full_drum_audio, y[int(initial_start_time * sr):] / 2, stereo=True)

    return full_drum_audio, full_tap_audio, tap_and_onsets, drums_and_original, combined_drum_sequence"""


def midi_to_wav(data, sample_rate, sf2_path=None):
    wav_data = librosa.util.normalize(midi_synth.fluidsynth(data, sample_rate=sample_rate, sf2_path=sf2_path))
    return wav_data
