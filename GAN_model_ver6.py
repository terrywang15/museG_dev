#!/usr/bin/env python
# coding: utf-8

# # MScA Capstone
# # GAN Model for Generating Music Samples with Midi Data
# 
# ## Josh Goldberg, Rima Mittal, Terry Wang

# Some sources considered for this version:
# https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/
# https://github.com/soumith/ganhacks
# Generative Deep Learning by David Foster

# In[16]:


import time
import os
import numpy as np
import pandas as pd
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, concatenate, BatchNormalization, Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout, Softmax, LeakyReLU, ReLU
from pretty_midi_musGen import PrettyMIDI, Instrument, Note
# Note: need to install package mido before running this script
# run this after running the model on the folder containing run files in case of need:
# sudo chown -R [user name] [folder_name]
# makes the user the owner of the run files

# # Model v6

# Issues from v5
# 1) Loss from discriminator quickly goes to near 0
# 2) Generator fails to convert loss into real learning
# 3) Loss generally is too small to drive updates

# Changes:
# 1) New model architecture with with separate pitch and duration generators and discriminators
#       to keep loss separate

# Future Changes:
# 1) Weissenstein Loss instead of binary crossentropy
# 2) Refine input data to only include one composer (Bach?)
# 3) Change duration encoding to (start time, duration)
# 4) Change generator structure, have dedicated pitch and dur generators each with own discriminator and loss
# 5) Can potentially adopt LSTM structure for the generator(s)


# Discriminator
def define_pitch_discriminator(in_shape=(20, 128)):
    """
    Outputs a model for pitch discriminator
    :param in_shape: shape of input
    :return: a keras model
    """
    # Sequence: Dense, BatchNorm, Activation, Dropout
    # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
    model = Sequential()
    model.add(Input(in_shape))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    # model.add(Dense(256))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.1))
    # model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def define_duration_discriminator(in_shape=(20, 2)):
    """
    Outputs a model for duration discriminator
    :param in_shape: shape of input
    :return: a keras model
    """
    # Sequence: Dense, BatchNorm, Activation, Dropout
    # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
    model = Sequential()
    model.add(Input(in_shape))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    # model.add(Dense(256))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.1))
    # model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


# Generator
def define_pitch_generator(latent_dim, dropout_rate, num_nodes):
    """
    Outputs a pitch generator model
    :param latent_dim: Int, dimension of latent points aka how many random numbers you want to use as input
    :param dropout_rate: float between 0 and 1, dropout rate you give to the dropout layers
    :param num_nodes: Int, number of nodes in the dense layers
    :return: a keras model (output shape 20, 128)
    """

    random_inputs = Input(shape=(latent_dim,))
    dense1 = Dense(num_nodes)(random_inputs)
    batchNorm1 = BatchNormalization()(dense1)
    activ1 = ReLU()(batchNorm1)
    dropout1 = Dropout(dropout_rate)(activ1)
    dense2 = Dense(num_nodes)(dropout1)
    batchNorm2 = BatchNormalization()(dense2)
    activ2 = ReLU()(batchNorm2)
    dropout2 = Dropout(dropout_rate)(activ2)

    pitch1 = Dense(num_nodes)(dropout2)
    pitch_bnorm1 = BatchNormalization()(pitch1)
    pitch_activ1 = ReLU()(pitch_bnorm1)
    pitch_dropoff1 = Dropout(dropout_rate)(pitch_activ1)
    pitch2 = Dense(num_nodes)(pitch_dropoff1)
    pitch_bnorm2 = BatchNormalization()(pitch2)
    pitch_activ2 = ReLU()(pitch_bnorm2)
    pitch_dropoff2 = Dropout(dropout_rate)(pitch_activ2)
    pitch = Dense(20 * 128)(pitch_dropoff2)

    pitch_reshaped = Reshape((20, 128))(pitch)

    pitch_output = Softmax(axis=-1)(pitch_reshaped)

    generator = Model(inputs=random_inputs, outputs=pitch_output)

    return generator


def define_duration_generator(latent_dim, dropout_rate, num_nodes):
    """
    Outputs a duration generator model
    :param latent_dim: Int, dimension of latent points aka how many random numbers you want to use as input
    :param dropout_rate: float between 0 and 1, dropout rate you give to the dropout layers
    :param num_nodes: Int, number of nodes in the dense layers
    :return: a keras model (output shape 20, 2)
    """

    random_inputs = Input(shape=(latent_dim,))
    dense1 = Dense(num_nodes)(random_inputs)
    batchNorm1 = BatchNormalization()(dense1)
    activ1 = ReLU()(batchNorm1)
    dropout1 = Dropout(dropout_rate)(activ1)
    dense2 = Dense(num_nodes)(dropout1)
    batchNorm2 = BatchNormalization()(dense2)
    activ2 = ReLU()(batchNorm2)
    dropout2 = Dropout(dropout_rate)(activ2)

    duration1 = Dense(num_nodes)(dropout2)
    dur_bnorm1 = BatchNormalization()(duration1)
    dur_activ1 = ReLU()(dur_bnorm1)
    dur_dropoff1 = Dropout(dropout_rate)(dur_activ1)
    duration2 = Dense(num_nodes)(dur_dropoff1)
    dur_bnorm2 = BatchNormalization()(duration2)
    dur_activ2 = ReLU()(dur_bnorm2)
    dur_dropoff2 = Dropout(dropout_rate)(dur_activ2)
    duration = Dense(20 * 2)(dur_dropoff2)

    duration_reshaped = Reshape((20, 2))(duration)
    duration_output = Dense(2, activation='relu', name='duration')(duration_reshaped)

    generator = Model(inputs=random_inputs, outputs=duration_output)

    return generator


# Adversarial Model
def define_gan(generator, discriminator):
    """
    Outputs a GAN model
    :param generator: the generator model
    :param discriminator: the discriminator model
    :return: a keras model
    """
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# # Helper functions
def generate_real_samples(dataset, n_samples, label_smoothing, kind):
    """
    Makes a batch of real 20-note music samples
    :param dataset: the dataset containing 20-note music samples (x, 20, 130)
    :param n_samples: number of samples to generate
    :param label_smoothing: boolean, if true, real samples labeled as slightly less than 1 (0.9 for now)
    :param kind: string, either "pitch" or "duration"
    :return: training samples and target variable (set as 1 for real samples)
    """
    ix = randint(0, dataset.shape[0], n_samples)
    if kind == "pitch":
        # take last 128 columns for pitch
        X = dataset[ix][:, :, 2:]
        if label_smoothing:
            # add stochastic noise
            y = np.ones((n_samples, 1)) - np.random.uniform(0.0, 0.1, (n_samples, 1))
        else:
            y = np.ones((n_samples, 1))
        return X, y
    elif kind == "duration":
        X = dataset[ix][:, :, :2]
        if label_smoothing:
            # add stochastic noise
            y = np.ones((n_samples, 1)) - np.random.uniform(0.0, 0.1, (n_samples, 1))
        else:
            y = np.ones((n_samples, 1))
        return X, y
    elif kind == "both":
        X_pitch = dataset[ix][:, :, 2:]
        X_dur = dataset[ix][:, :, :2]
        if label_smoothing:
            # add stochastic noise
            y = np.ones((n_samples, 1)) - np.random.uniform(0.0, 0.1, (n_samples, 1))
        else:
            y = np.ones((n_samples, 1))
        return X_dur, X_pitch, y


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


def generate_fake_samples(pitch_generator, dur_generator, latent_dim, n_samples, kind):
    """
    makes fake samples using the generator
    :param generator: a keras model
    :param latent_dim: dimension of latent points, need to be the same as the generator's
    :param n_samples: number of samples you want to generate
    :param kind: str, "concat" or "separate"
    :return: training samples and target variable (set as 0 for fake samples)
    """
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    if kind == "concat":
        # predict fake music
        duration = dur_generator.predict(x_input)
        pitches = pitch_generator.predict(x_input)
        pitches = np.eye(128)[np.argmax(pitches, axis=-1)]
        X = np.concatenate((duration, pitches), axis=-1)
        y = np.zeros((n_samples, 1)) # label them as zeros
        return X, y
    elif kind == "separate":
        # predict fake music
        duration = dur_generator.predict(x_input)
        pitches = pitch_generator.predict(x_input)
        pitches = np.eye(128)[np.argmax(pitches, axis=-1)]
        y = np.zeros((n_samples, 1))
        return duration, pitches, y


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


def make_assembled_pred(pitch_gen, dur_gen, latent_dim, n_samples):
    """
    Make a prediction from assembled outputs of pitch and duration generators
    :param pitch_gen: keras model, pitch generator
    :param dur_gen: keras model, duration generator
    :param latent_dim: int, number of latent dims needed for both generators
    :param n_samples: int, number of samples to generate
    :return: a numpy array
    """

    # Get latent points
    x_input = generate_latent_points(latent_dim, n_samples)
    # Generate pitch and duration
    duration = dur_gen.predict(x_input)
    pitch = pitch_gen.predict(x_input)
    # Take the highest value for pitch
    pitch = np.eye(128)[np.argmax(pitch, axis=-1)]
    # concat for output
    pred = np.concatenate((duration, pitch), axis=-1)

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


# # Model Training
def train(g_pitch_model, g_dur_model, d_pitch_model, d_dur_model, gan_pitch_model, gan_dur_model, dataset, latent_dim,
          n_epochs=100, n_batch=128):
    """
    GAN training script
    :param g_pitch_model: keras model, pitch generator
    :param g_dur_model: keras model, duration generator
    :param d_pitch_model: keras model, pitch discriminator
    :param d_dur_model: keras model, duration discriminator
    :param gan_pitch_model: keras model, pitch GAN
    :param gan_dur_model: keras model, duration GAN
    :param dataset: numpy array, real samples
    :param latent_dim: int, latent dim as generator input
    :param n_epochs: int, number of epochs for train
    :param n_batch: int, number of batches per epoch. This actually is the number of real samples fed into the model
    :return: nothing

    Also note, n_batch also controls how many real/fake samples will be seen per epoch.
    The way the code is written, the discriminator will see n_batch number of samples per epoch, half fake and half real
    GAN model will receive 2 times n_batch number of samples per epoch
    So you will not see all of the data per epoch (the usual definition of an epoch)
    The name can be a bit confusing there
    """

    # generator model weights, sample output and loss history saved to timestamped folder
    # dataframe for recording history
    df_history = pd.DataFrame(columns=["epoch", "batch", "d_dur_loss_real", "d_dur_loss_fake", "d_pitch_loss_real",
                                       "d_pitch_loss_fake", "g_dur_loss", "g_pitch_loss"])
    df_history.to_csv(ts_str + '/loss_history.csv', index=False)
    # batches per epoch
    bat_per_epo = int(dataset.shape[0] / n_batch)
    real_batch = int(n_batch / 2)
    fake_batch = n_batch - real_batch
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get half_batch randomly selected real samples
            X_dur_real, X_pitch_real, y_real = generate_real_samples(dataset, real_batch, False, "both")
            # update weights of discriminator model on real duration samples
            d_dur_loss1, _ = d_dur_model.train_on_batch(X_dur_real, y_real)
            # update weights of discriminator model on real pitch samples
            d_pitch_loss1, _ = d_pitch_model.train_on_batch(X_pitch_real, y_real)
            # generate fake samples
            X_dur_fake, X_pitch_fake, y_fake = generate_fake_samples(g_pitch_model, g_dur_model, latent_dim, fake_batch,
                                                                     "separate")
            # update discriminator model weights on fake samples
            d_dur_loss2, _ = d_dur_model.train_on_batch(X_dur_fake, y_fake)
            d_pitch_loss2, _ = d_pitch_model.train_on_batch(X_pitch_fake, y_fake)
            # train GAN
            for _ in range(2):
                # prepare points in latent space as input for the generator
                # In this case we strictly use the same latent points for both dur and pitch
                X_gan = generate_latent_points(latent_dim, n_batch)
                # label fake samples as ones to train generator
                y_gan = np.ones((n_batch, 1))
                # update weights of the generator based on the discriminator's errors
                g_dur_loss = gan_dur_model.train_on_batch(X_gan, y_gan)
                g_pitch_loss = gan_pitch_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d_dur_1= %.3f, d_dur_2= %.3f, d_pit_1= %.3f, d_pit_2= %.3f, g_dur= %.3f, g_pit= %.3f' %
                  (i + 1, j + 1, bat_per_epo, d_dur_loss1, d_dur_loss2, d_pitch_loss1, d_pitch_loss2, g_dur_loss,
                   g_pitch_loss))
            df_history.loc[len(df_history)] = [i + 1, j + 1, d_dur_loss1, d_dur_loss2, d_pitch_loss1, d_pitch_loss2,
                                               g_dur_loss, g_pitch_loss]
        # Generate a midi sample every 100 epochs
        if i % 50 == 0:
            pred = make_assembled_pred(g_pitch_model, g_dur_model, latent_dim, 1)
            make_midi(pred, ts_str + "/ep_" + str(i) + ".mid")

            # append to history
            df_history.to_csv(ts_str + '/loss_history.csv', mode='a', index=False, header=False)
            df_history = pd.DataFrame(columns=["epoch", "batch", "d_dur_loss_real", "d_dur_loss_fake",
                                               "d_pitch_loss_real", "d_pitch_loss_fake", "g_dur_loss", "g_pitch_loss"])
            # save the model
            g_dur_model.save(ts_str + '/g_dur_wgts_ep_' + str(i) + '.h5')
            g_pitch_model.save(ts_str + '/g_pitch_wgts_ep_' + str(i) + '.h5')
    # save the final generator model
    g_dur_model.save(ts_str + '/g_dur_wgts_ep_' + str(i) + '.h5')
    g_pitch_model.save(ts_str + '/g_pitch_wgts_ep_' + str(i) + '.h5')
    # save a final sample
    pred = make_assembled_pred(g_pitch_model, g_dur_model, latent_dim, 1)
    make_midi(pred, ts_str + "/ep_" + str(i) + ".mid")
    # save loss history for analysis
    df_history.to_csv(ts_str + '/loss_history.csv', mode='a', index=False, header=False)

## TRAINING PARAMETERS ##

# size of the latent space
latent_dim = 128
# number of nodes in dense layers
num_nodes = 256
# generator dropout
gen_dropout = 0.2
# other parameters
n_epochs = 1000
n_batch = 600
# create the discriminator
dur_discriminator = define_duration_discriminator()
pitch_discriminator = define_pitch_discriminator()
# create the generators
dur_generator = define_duration_generator(latent_dim, gen_dropout, num_nodes)
pitch_generator = define_pitch_generator(latent_dim, gen_dropout, num_nodes)
# create the gan
dur_gan_model = define_gan(dur_generator, dur_discriminator)
pitch_gan_model = define_gan(pitch_generator, pitch_discriminator)
# load real music sample data
train_fp = "training_samples/sample_2020-02-13-17-14.npy"
dataset = np.load(train_fp)
# Make timestamp
ts_str = time.strftime("%Y-%m-%d %H-%M", time.gmtime())
# create directory if not exist
if not os.path.exists(ts_str):
    os.makedirs(ts_str)
# create log for training metadata
with open("metadata.txt", "w+") as f:
    f.write("training script: GAN_model_ver" + "6" + ".py\n")
    f.write("training sample: " + train_fp)
    f.write("latent dim: %i \n" % latent_dim)
    f.write("number of nodes : %i \n" % num_nodes)
    f.write("generator dropout rate: %f.2 \n" % gen_dropout)
    f.write("number of epochs: %i \n" % n_epochs)
    f.write("samples per epoch: %i \n" % n_batch)

# In[18]:


# train model
train(pitch_generator, dur_generator, pitch_discriminator, dur_discriminator, pitch_gan_model, dur_gan_model, dataset,
      latent_dim, n_epochs=n_epochs, n_batch=n_batch)





