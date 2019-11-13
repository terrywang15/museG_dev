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
    np_midi = np.load(np_midi_file)

    # remove track if sample length is greater than the track's length
    tracks_to_use = []
    for idx in range(len(np_midi)):
        if np_midi[idx].shape[0] >= sample_length:
            tracks_to_use.append(idx)

    num_sampled = 0

    while num_sampled < n_samples:
        # first, randomly sample a track
        samp_track = np_midi[random.randint(0, len(tracks_to_use)-1)]

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