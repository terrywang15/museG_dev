# museG_dev
Repo with original and forked (from pretty_midi) scripts for easy midi preprocessing for deep learning.  

The purpose of the scripts (`write_midi_as_np.py` and `generate_training_samples.py`) are to encode midi files as numpy arrays and to generate arbitrary number of note sequences of any length from the midi files encoded as numpy arrays, respectively.  Therefore, you must run `write_midi_as_np.py` on your folder of midi files before you can run `generate_training_samples.py`.

## How to use
Install as you would any git repo, but you must also install all dependencies of `pretty_midi`.

### `write_midi_as_np.py`
Use this to encode midi files into numpy arrays.  During the process we will only retain the following information: note start time, end time, and pitch.  

The encoded file will have a shape of (`num_tracks`, ) where `num_tracks` refers to the number of tracks present in the original midi file.  However, each track will also be a numpy array of shape (`num_notes`, 3) where `num_notes` refers to the number of notes present in the track.  The number 3 refers to the information we retain from the original midi file: start time, end time, pitch.

You must specify `source_filepath`, `target_filepath`, `target_key`, and `max_n_files` as arguments to run the file.  

`source_filepath` is the path to the FOLDER containing midi files.

`target_filepath` is the path to the FOLDER where you want the np arrays to be.

`target_key` is the key you want the songs to be converted to, and must be a number between 0 and 11. If you do not wish to convert keys, then pass `None`.

`max_n_files` is the max number of files you want processed.  If you wish to process all files in the folder, then pass `None`.

Example: `python midi_files midi_np_files None None`

### `generate_training_samples.py`
Run this script to generate note sequence samples for deep learning, with pitches being converted into one-hot.  You can specify how long each note sequence should be (`sample_length`) and how many samples to generate (`n_samples`).  The output will have shape (`n_samples`, `sample_length`, 130) where 130 refers to start time and end time, plus the one-hot encoded pitches (length of 128).

Example: `python midi_np_files training_samples 20 50000`
