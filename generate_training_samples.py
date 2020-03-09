import glob
import datetime
import time
import sys
import os
import random

import numpy as np

from pretty_midi_musGen.musGen_utils import generate_samples, sample_pitch_to_onehot

# sample usage
# python3 generate_training_samples.py midi_np_data/bach_choral training_samples 20 100000 False

def main(source_folder, samples_folder, sample_length, n_samples, end_time):
    """
    generate samples from folder containing .npy files, with specified length and num of samples
    :param source_folder: folder containing .npy files
    :param samples_folder: target folder to write to
    :param sample_length: how many notes should each sample contain
    :param n_samples: how many samples should we generate
    :param end_time: str, "True" then second column indicates end timestamp
    :return: nothing - samples written as .npy file to samples folder with a .txt log

    ## Note: .npy files must exist directly under source folder and not under sub folders
    """

    output = []

    # create directory if not exist
    if not os.path.exists(samples_folder):
        os.makedirs(samples_folder)

    # iterable of all .npy files
    all_files = glob.glob(source_folder + '/*.npy')
    num_files = len(all_files)
    # print(num_files)

    # record which files were processed/unprocessed
    processed_files = []
    unprocessed_files = []

    # check whether ending time is end time or note length
    end_time_ind = False
    if end_time == "True":
        end_time_ind = True

    num_sampled = 0
    while num_sampled < n_samples:
        # randomly pick one file
        sample_file = all_files[random.randint(0, num_files-1)]
        # get 1 random sample from that file
        try:
            samp = generate_samples(sample_file, sample_length, 1, end_time_ind)
            processed_files.append(sample_file)
            # print("successfully processed " + sample_file)
            output.append(sample_pitch_to_onehot(samp))
            num_sampled += 1
        except Exception as e:
            print(e)
            unprocessed_files.append(sample_file)

    timestamp = time.time()
    st = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H-%M')
    np.save(samples_folder+'/'+'sample_'+st+'.npy', np.array(output).reshape((n_samples, sample_length, 130)))

    processed_files = list(set(processed_files))

    print("sampled from %d files" % len(processed_files))
    print("generated %d samples" % num_sampled)

    with open(samples_folder+'/'+'run_log_'+st+'.txt', 'w') as f:
        for pf in processed_files:
            f.write(pf+'\n')

    with open(samples_folder+'/'+'run_log_unprocessed'+st+'.txt', 'w') as f:
        for pf in processed_files:
            f.write(pf+'\n')


if __name__ == '__main__':
    sys.exit(main(source_folder=sys.argv[1],
                  samples_folder=sys.argv[2],
                  sample_length=int(sys.argv[3]),
                  n_samples=int(sys.argv[4]),
                  end_time=sys.argv[5]))
