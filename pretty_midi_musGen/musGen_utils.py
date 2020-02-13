"""
Utility functions for handling MIDI data in an easy to read/manipulate
format for the Capstone project
"""
# from __future__ import print_function

import random
import numpy as np

from .pretty_midi import PrettyMIDI
from .instrument import Instrument
from .containers import Note


def getDataFromMidi(midi_file, target_key):
    """
    convert midi file to np array akin to a list of numpy arrays
    inputs:
        midi_file: midi file
        target_key: if not None, will convert to target key [0-11]
    outputs:
        list of np arrays with 3 columns (depending on the number of tracks)
    """
    # Note: Using forked version of pretty_midi during implementation
    # Note: remember to turn on ignore tempo changes
    midi_data = PrettyMIDI(midi_file, ignore_tempo_changes=True)

    output_tracks = []
    # if a target key is passed then will attempt to convert to that key
    # take the initial key as the song's key
    # If no initial key, throw an exception
    if target_key is not None:
        try:
            original_key = midi_data.key_signature_changes[0].key_number
        except IndexError as e:
            raise "invalid key"

    # make one np array for each track, loop through them
    for track in midi_data.instruments:
        # ignore drums
        if track.is_drum:
            continue
        # Here we check whether the track is a "melody" track
        # which is something we will have to define
        # for now I am making anything related to piano/violin/cello/guitar to be melody tracks
        # But this can change as we iterate over the work
        # Will use program number for this task
        # filter out bass instruments
        if track.program in [33, 34, 35, 36, 37, 38, 39, 40, 44, 59]:
            continue
            # some other cases such as sparcity of notes can indicate that the track is complementary
            # can be thrown out as well
        # rewrite notes to np.array, but first sort by start time
        track_sorted = sorted(track.notes, key=lambda x: x.start)
        midi_np = np.array([[note.start, note.end, note.pitch] for note in track_sorted])

        # Try converting to target key
        if target_key is not None:
            output_tracks.append(keyConvert(midi_np, original_key, target_key))
        else:
            output_tracks.append(midi_np)

    return np.array(output_tracks)


def getDataFromMidi2(midi_file, target_key):
    """
    convert midi file to np array akin to a list of numpy arrays
    with duration encoded as (start time, duration)
    inputs:
        midi_file: midi file
        target_key: if not None, will convert to target key [0-11]
    outputs:
        list of np arrays with 3 columns (depending on the number of tracks)
    """
    # Note: Using forked version of pretty_midi during implementation
    # Note: remember to turn on ignore tempo changes
    midi_data = PrettyMIDI(midi_file, ignore_tempo_changes=True)

    output_tracks = []
    # if a target key is passed then will attempt to convert to that key
    # take the initial key as the song's key
    # If no initial key, throw an exception
    if target_key is not None:
        try:
            original_key = midi_data.key_signature_changes[0].key_number
        except IndexError as e:
            raise "invalid key"

    # make one np array for each track, loop through them
    for track in midi_data.instruments:
        # ignore drums
        if track.is_drum:
            continue
        # Here we check whether the track is a "melody" track
        # which is something we will have to define
        # for now I am making anything related to piano/violin/cello/guitar to be melody tracks
        # But this can change as we iterate over the work
        # Will use program number for this task
        # filter out bass instruments
        if track.program in [33, 34, 35, 36, 37, 38, 39, 40, 44, 59]:
            continue
            # some other cases such as sparcity of notes can indicate that the track is complementary
            # can be thrown out as well
        # rewrite notes to np.array, but first sort by start time
        track_sorted = sorted(track.notes, key=lambda x: x.start)
        midi_np = np.array([[note.start, note.end - note.start, note.pitch] for note in track_sorted])

        # Try converting to target key
        if target_key is not None:
            output_tracks.append(keyConvert(midi_np, original_key, target_key))
        else:
            output_tracks.append(midi_np)

    return np.array(output_tracks)


def toMajorKey(key):
    """
    make all key signatures into its major equivalent
    inputs:
        key: number between [0-23]
    outputs:
        output_key: number between [0-11]
    """

    if key >= 0 and key <= 11:
        return key
    elif key >= 12 and key <= 23:
        return (key % 12 + 3) % 12


def keyConvert(NP, original_key, target_key):
    """
    convert midi in numpy array format to a different key
    inputs:
        NP: midi file in np.array form
        original_key: original key of NP file [0-23]
        target_key: key to convert to [0-23]
    output:
        a midi_np with key converted
    """
    # Maybe check if keys are out of range
    # first convert original key and target key to major equivalents
    orig_key = toMajorKey(original_key)
    targ_key = toMajorKey(target_key)

    if orig_key == targ_key:
        return NP
    else:
        # idea is convert to 0 (C major)
        # if difference is less than or equal to 6, move NP up or down directly
        # else, move NP up or down in reverse direction
        # hopefully this minimizes the chance of notes going out of bounds (0-127)
        # if they do, just convert to one octive above (0) or below (127)
        key_diff = target_key - original_key

        if abs(key_diff) <= 6:
            return NP + [0, 0, key_diff]
        elif abs(key_diff) > 6 and abs(key_diff) <= 11:
            if key_diff < 0:
                np_conv = NP + [0, 0, key_diff + 12]
                # check if pitch is out of bounds
                if sum(np_conv[:, 2] > 127) or sum(np_conv[:, 2] < 0):
                    # replace rows > 127 with pitch 12 tones lower
                    np_conv[np_conv[:, 2] > 127, 2] = np_conv[np_conv[:, 2] > 127, 2] - 12
                    # replace rows < 0 with pitch 12 tones higher
                    np_conv[np_conv[:, 2] < 0, 2] = np_conv[np_conv[:, 2] < 0, 2] + 12
                    return np_conv
                else:
                    return np_conv
            elif key_diff > 0:
                np_conv = NP + [0, 0, key_diff - 12]
                # check if pitch is out of bounds
                if sum(np_conv[:, 2] > 127) or sum(np_conv[:, 2] < 0):
                    # replace rows > 127 with pitch 12 tones lower
                    np_conv[np_conv[:, 2] > 127, 2] = np_conv[np_conv[:, 2] > 127, 2] - 12
                    # replace rows < 0 with pitch 12 tones higher
                    np_conv[np_conv[:, 2] < 0, 2] = np_conv[np_conv[:, 2] < 0, 2] + 12
                    return np_conv
                else:
                    return np_conv


def generate_samples(np_midi_file, sample_length, n_samples=1):
    """
    Generate random samples of a certain length from a midi file in numpy form
    :param np_midi_file: a .npy file which can be understood as a list of numpy arrays
    :param sample_length: number of notes contained in each sample generated
    :param n_samples: number of samples to be generated
    :return: a numpy array of shape (n_samples, sample_length, 3)
    """

    output = []

    # load the file
    # new in numpy 1.17.1: must set allow pickle to true, but beware of security risk as you are loading a binary
    np_midi = np.load(np_midi_file, allow_pickle=True)

    # remove track if sample length is greater than the track's length
    tracks_to_use = []
    for idx in range(len(np_midi)):
        if np_midi[idx].shape[0] >= sample_length:
            tracks_to_use.append(idx)

    num_sampled = 0

    while num_sampled < n_samples:
        # first, randomly sample a track
        samp_track = np_midi[tracks_to_use[random.randint(0, len(tracks_to_use)-1)]]

        # construct valid range for random sampling of starting positions
        max_pos = samp_track.shape[0] - sample_length

        # sample a starting position
        samp_startPos = random.randint(0, max_pos)

        # make the sample
        samp_notes = samp_track[samp_startPos:samp_startPos+sample_length, :]

        # normalize start time to 0
        samp_notes = samp_notes - [samp_notes[0, 0], samp_notes[0, 0], 0]

        output.append(samp_notes)

        num_sampled += 1

    return np.array(output)


def sample_pitch_to_onehot(np_midi_file):
    """
    Converts pitch column from number to one-hot of array of 128 elements
    This only works with data shape (x, y, 3) where
    x is number of samples
    y is number of notes per sample
    3 refers to the 3 columns from midi file (start time, end time, pitch)
    See to_onehot function for modularized version
    :param np_midi_file: an array (x, y, 3) with 3 columns, the last of which is the pitch column
    :return: an np array wtih 130 (2+128) columns with pitch one-hot encoded
    """

    # Create x, y, 128 array template for pitches
    num_samps = np_midi_file.shape[0]
    num_notes = np_midi_file.shape[1]
    output = []
    for samp_idx in range(num_samps):
        # Create x, y, 128 array template for pitches
        template = np.zeros((num_notes, 128))
        # Replace 0 as 1 at the right places
        template[np.arange(num_notes), np_midi_file[samp_idx, :, -1].astype(int)] = 1.
        # Concat with real data and append
        output.append(np.concatenate((np_midi_file[samp_idx, :, :-1], template), axis = 1))

    # Concat and return output so that shape will remain the same for the first two axes
    return np.concatenate(output, axis=0)


def pitch_to_onehot(array_of_pitches):
    """
    generalized and modularized pitch to one hot function
    converts array of pitch numbers into a one hot matrix
    :param array_of_pitches: flat array of integers between 0 and 127
    :return: matrix of shape (len(array_of_pitches), 128)
    """

    num_notes = len(array_of_pitches)
    # create template
    template = np.zeros((num_notes, 128))
    # fill template with 1 where applicable
    template[np.arange(num_notes), array_of_pitches.astype(int)] = 1.

    return template


def onehot_to_pitch(matrix_of_ohe_pitches):
    """
    converts matrix of one hot encoded pitches to integer
    inverse of to_onehot
    :param matrix_of_ohe_pitches: matrix of shape (num_notes, 128)
    :return: an array of length equal to num_notes
    """

    return np.where(matrix_of_ohe_pitches == 1.)[1]


def midi2NP(midi_file):
    """
    randomly selects
    :param midi_file:
    :return:
    """

    music = PrettyMIDI(midi_file, ignore_tempo_changes=True)
    output = []
    for track in music.instruments:
        track_output = []
        for note in track.notes:
            track_output.append([note.start, note.end, note.pitch])
        track_output = np.array(track_output)
        output.append(track_output)

    return output


def notes2Midi(notes, output_file_path):
    start_time = notes[0][:, 0].item()
    # create PrettyMIDI object
    output = PrettyMIDI()
    # Use piano as the instrument
    instr = Instrument(0, name = 'Piano')
    # loop over notes
    for note in notes:
        flattened_note = note.flatten()
        note_to_add = Note(velocity=100, pitch=int(flattened_note[2]), start=flattened_note[0]-start_time, end=flattened_note[1]-start_time)
        instr.notes.append(note_to_add)
    output.instruments.append(instr)
    output.write(output_file_path)


def make_midi(pred, write_file_name):
    """
    Makes a midi file from a numpy array - for duration encoded as (start time, end time)
    :param pred: a numpy array of shape (1, 20, 130) (can only process one sample at a time)
    :param write_file_name: location and name of the file you want to write
    :return: nothing, writes a midi file to specified location
    """
    midi = PrettyMIDI()
    notes = np.argmax(pred[0, :, 2:], axis=-1)
    instr = Instrument(0, name='Piano')
    for idx, note in enumerate(notes):
        note_to_add = Note(velocity=100, pitch=int(note), start=min(pred[0, idx, 0], pred[0, idx, 1]),
                           end=max(pred[0, idx, 0], pred[0, idx, 1]))
        instr.notes.append(note_to_add)
    midi.instruments.append(instr)
    midi.write(write_file_name)


def make_midi2(pred, write_file_name):
    """
    Makes a midi file from a numpy array - for duration encoded as (start time, duration)
    :param pred: a numpy array of shape (1, 20, 130) (can only process one sample at a time)
    :param write_file_name: location and name of the file you want to write
    :return: nothing, writes a midi file to specified location
    """
    midi = PrettyMIDI()
    notes = np.argmax(pred[0, :, 2:], axis=-1)
    instr = Instrument(0, name='Piano')
    for idx, note in enumerate(notes):
        note_to_add = Note(velocity=100, pitch=int(note), start=pred[0, idx, 0],
                           end=pred[0, idx, 0] + pred[0, idx, 1])
        instr.notes.append(note_to_add)
    midi.instruments.append(instr)
    midi.write(write_file_name)