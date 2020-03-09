"""Collection of training utils"""

import numpy as np
from numpy.random import randn
from numpy.random import randint
from pretty_midi_musGen import PrettyMIDI, Instrument, Note


def generate_real_samples(dataset, n_samples):
    """
    Makes a batch of real 20-note music samples
    :param dataset: the dataset containing 20-note music samples (x, 20, 130)
    :param n_samples: number of samples to generate
    :return: training samples and target variable (set as 1 for real samples)
    """
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1)) # labeled as 1
    return X, y


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


def generate_fake_samples(generator, latent_dim, n_samples):
    """
    makes fake samples using the generator
    :param generator: a keras model
    :param latent_dim: dimension of latent points, need to be the same as the generator's
    :param n_samples: number of samples you want to generate
    :return: training samples and target variable (set as 0 for fake samples)
    """
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict fake music
    X = generator.predict(x_input)
    duration = X[:,:,:2]
    pitches = np.eye(128)[np.argmax(X[:,:,2:], axis=-1)]
    X = np.concatenate((duration, pitches), axis=-1)
    y = np.zeros((n_samples, 1)) # label them as zeros
    return X, y


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