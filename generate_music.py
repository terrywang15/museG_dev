import time
import os
import numpy as np
from numpy.random import randn
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, concatenate, BatchNormalization, Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout, Softmax
from pretty_midi_musGen import PrettyMIDI, Instrument, Note


# Generator
def define_generator(latent_dim, dropout_rate, num_nodes):
    """
    Outputs a generator model
    :param latent_dim: Int, dimension of latent points aka how many random numbers you want to use as input
    :param dropout_rate: float between 0 and 1, dropout rate you give to the dropout layers
    :param num_nodes: Int, number of nodes in the dense layers
    :return: a keras model
    """

    random_inputs = Input(shape=(latent_dim,))
    dense1 = Dense(num_nodes, activation='relu')(random_inputs)
    dropout1 = Dropout(dropout_rate)(dense1)
    batchNorm1 = BatchNormalization()(dropout1)
    dense2 = Dense(num_nodes, activation='relu')(batchNorm1)
    dropout2 = Dropout(dropout_rate)(dense2)
    batchNorm2 = BatchNormalization()(dropout2)
    dense3 = Dense(num_nodes, activation='relu')(batchNorm2)
    dropout3 = Dropout(dropout_rate)(dense3)
    batchNorm3 = BatchNormalization()(dropout3)
    dense4 = Dense(num_nodes, activation='relu')(batchNorm3)
    dropout4 = Dropout(dropout_rate)(dense4)
    batchNorm4 = BatchNormalization()(dropout4)
    pitch1 = Dense(num_nodes, activation='relu')(batchNorm4)
    pitch_dropoff1 = Dropout(dropout_rate)(pitch1)
    duration1 = Dense(num_nodes, activation='relu')(batchNorm4)
    dur_dropoff1 = Dropout(dropout_rate)(duration1)
    pitch2 = Dense(num_nodes, activation='relu')(pitch_dropoff1)
    duration2 = Dense(num_nodes, activation='relu')(dur_dropoff1)
    pitch = Dense(20 * 128)(pitch2)
    duration = Dense(20 * 2)(duration2)

    pitch_reshaped = Reshape((20, 128))(pitch)

    duration_reshaped = Reshape((20, 2))(duration)

    pitch_output = Softmax(axis=-1)(pitch_reshaped)

    duration_output = Dense(2, activation='relu', name='duration')(duration_reshaped)
    output_concat = concatenate([duration_output, pitch_output])
    generator = Model(inputs=random_inputs, outputs=output_concat)

    return generator


def generate_latent_points(latent_dim, n_samples):
    """
    makes arrays of random numbers of the right shape for the generator
    :param latent_dim: dimension of latent points, need to be the same as the generator's
    :param n_samples: number of samples you want to generate
    :return: a numpy array
    """
    # 'n_samples' vectors of 'latent_dim' standard normal numbers each
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def make_pred(generator, latent_dim, num_samples):
    """
    Makes a prediction using the generator
    :param generator: a keras model
    :param latent_dim: dimension of latent points, need to be the same as the generator's
    :param num_samples: number of samples you want to generate
    :return: a numpy array
    """

    gen_sample = generator.predict(generate_latent_points(latent_dim, num_samples))
    duration = gen_sample[:,:,:2]
    pitches = np.eye(128)[np.argmax(gen_sample[:,:,2:], axis=-1)]
    pred = np.concatenate((duration, pitches), axis=-1)

    return pred


def make_midi(pred, write_file_name):
    """
    Makes a midi file from a numpy array
    :param pred: a numpy array of shape (1, 20, 130) (can only process one sample at a time)
    :param write_file_name: location and name of the file you want to write
    :return: nothing, writes a midi file to specified location
    """
    midi = PrettyMIDI()
    notes = np.argmax(pred[0,:,2:], axis=-1)
    instr = Instrument(0, name = 'Piano')
    for idx, note in enumerate(notes):
        note_to_add = Note(velocity=100, pitch=int(note), start=min(pred[0,idx,0], pred[0,idx,1]), end=max(pred[0,idx,0], pred[0,idx,1]))
        instr.notes.append(note_to_add)
    midi.instruments.append(instr)
    midi.write(write_file_name)


def make_multi_midi(pred, write_folder_name):
    """
    Makes a midi file from a numpy array
    :param pred: a numpy array of shape (x, 20, 130) (x is sample size)
    :param write_file_name: location and name of the file you want to write
    :return: nothing, writes a midi file to specified location
    """
    if not os.path.exists(write_folder_name):
        os.makedirs(write_folder_name)
    ts_str_mid = time.strftime("%Y-%m-%d %H-%M", time.gmtime())
    for idx, sample in enumerate(pred):
        midi = PrettyMIDI()
        notes = np.argmax(sample[:,2:], axis=-1)
        instr = Instrument(0, name='Piano')
        for idx, note in enumerate(notes):
            note_to_add = Note(velocity=100, pitch=int(note), start=min(sample[idx, 0], sample[idx, 1]),
                               end=max(sample[idx, 0], sample[idx, 1]))
            instr.notes.append(note_to_add)
        midi.instruments.append(instr)
        midi.write(write_folder_name + "/" + ts_str_mid + str(idx) + ".mid")


# Model parameters
# Number of files to generate
n_output = 1
# size of the latent space
latent_dim = 128
# length of noise vectors
num_nodes = 256
# generator dropout
gen_dropout = 0.2
# model filepath
weights_fp = "2020-02-12 19-31/gen_wgts_ep_700.h5"
# Make timestamp
ts_str = time.strftime("%Y-%m-%d %H-%M", time.gmtime())
# create directory if not exist
if not os.path.exists("gen_samples"):
    os.makedirs("gen_samples")
# output filepath
output_fp = "gen_samples/" + ts_str + ".mid"

model = define_generator(latent_dim, gen_dropout, num_nodes)
model.load_weights(weights_fp)
make_midi(make_pred(model, latent_dim, n_output), output_fp)

