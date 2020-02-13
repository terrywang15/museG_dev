import glob
import datetime
import time
import sys
import os

import numpy as np

from pretty_midi_musGen.musGen_utils import getDataFromMidi

# sample usage
# python3 write_midi_as_np.py "training_midis" "midi_np_data/bach_choral" "None" "None"

def main(source_filepath, target_filepath, target_key='None', max_n_files='None'):
    """
    Write all midi files into numpy array with key and tempo adjustment (in future release)
    :param source_filepath: folder containing midi files
    :param target_filepath: folder to write numpy arrays to
    :param target_key: target key to convert to, pass string 'None' if no conversion
    :param max_n_files: max number of files to be processed to avoid consuming too much space, pass 'None' for no limit
    :return: nothing - numpy files written to disk in loop to save memory
    in addition, write a .txt file with all processed files
    """

    # Check target key to make it correct
    if target_key == 'None':
        tkey = None
    else:
        tkey = int(target_key)

    # create directory if not exist
    if not os.path.exists(target_filepath):
        os.makedirs(target_filepath)

    # record which files were processed
    processed_files = []

    if max_n_files == 'None':
        for file in glob.iglob(source_filepath+'/**/*.mid'):
            in_fname = str(file)
            # print('processing ' + in_fname)
            out_fname = in_fname.split('/')[-1]
            try:
                np_midi = getDataFromMidi(file, tkey)
                np.save(target_filepath+'/'+out_fname+'.npy', np_midi)
                processed_files.append(in_fname)
                # print('processed ' + in_fname)
            except:
                print(in_fname+" unable to be processed, continuing")
                continue

    else:
        nfiles_processed = 0
        for file in glob.iglob(source_filepath+'/**/*.mid'):
            if nfiles_processed > int(max_n_files):
                break
            in_fname = str(file)
            out_fname = in_fname.split('/')[-1]
            try:
                np_midi = getDataFromMidi(file, tkey)
                np.save(target_filepath+'/'+out_fname+'.npy', np_midi)
                processed_files.append(in_fname)
                # print('processed ' + in_fname)
            except:
                print(in_fname+" unable to be processed, continuing")
                continue

    # write txt file for processed files
    timestamp = time.time()
    st = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H-%M')
    with open(target_filepath+'/'+'processed'+'_'+st+'.txt', 'w') as f:
        for pf in processed_files:
            f.write(pf+'\n')


if __name__ == '__main__':
    sys.exit(main(source_filepath=sys.argv[1],
                  target_filepath=sys.argv[2],
                  target_key=sys.argv[3],
                  max_n_files=sys.argv[4]))
