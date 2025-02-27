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
from tensorflow.keras.layers import Dropout, Softmax, ReLU, LeakyReLU
from pretty_midi_musGen import PrettyMIDI, Instrument, Note
# Note: need to install package mido before running this script
# run this after running the model on the folder containing run files in case of need:
# sudo chown -R [user name] [folder_name]
# makes the user the owner of the run files

# # Model v5

# Issues from v4
# 1) Fake loss seems to go up faster than real loss
# 2) Mode collapse occurred at epoch 555
# 3) Generally generator loss goes up with every epoch
# 4) Model errors quickly collapse to 0 in many situations

# Changes:
# 1) Add option to introduce noise to real image labels
# 2) Use LeakyReLU in discriminator
# 3) Additional BatchNorm layers in discriminator/generator
# 4) Save model weights every 100 epochs in a new file
# 5) Fixed order of batchnorm, activation, and dropout layers
# 6) Increased number of loops for generators to train per epoch from 2 to 5 - seems delay g-loss collapse

# Future Changes:
# 1) Weissenstein Loss instead of binary crossentropy
# 2) Refine input data to only include one composer (Bach?)
# 3) Change duration encoding to (start time, duration)
# 4) Change generator structure, have dedicated pitch and dur generators each with own discriminator and loss
# 5) Can potentially adopt LSTM structure for the generator(s)


# Discriminator
def define_discriminator(in_shape=(20, 130)):
    """
    Outputs a model for discriminator
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
    model.add(BatchNormalization())
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
def define_generator(latent_dim, dropout_rate, num_nodes):
    """
    Outputs a generator model
    :param latent_dim: Int, dimension of latent points aka how many random numbers you want to use as input
    :param dropout_rate: float between 0 and 1, dropout rate you give to the dropout layers
    :param num_nodes: Int, number of nodes in the dense layers
    :return: a keras model
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
    dense3 = Dense(num_nodes)(dropout2)
    batchNorm3 = BatchNormalization()(dense3)
    activ3 = ReLU()(batchNorm3)
    dropout3 = Dropout(dropout_rate)(activ3)
    dense4 = Dense(num_nodes)(dropout3)
    batchNorm4 = BatchNormalization()(dense4)
    activ4 = ReLU()(batchNorm4)
    dropout4 = Dropout(dropout_rate)(activ4)

    # divergent generators for pitch and duration
    pitch1 = Dense(num_nodes)(dropout4)
    pitch_bnorm1 = BatchNormalization()(pitch1)
    pitch_activ1 = ReLU()(pitch_bnorm1)
    pitch_dropoff1 = Dropout(dropout_rate)(pitch_activ1)
    duration1 = Dense(num_nodes)(dropout4)
    dur_bnorm1 = BatchNormalization()(duration1)
    dur_activ1 = ReLU()(dur_bnorm1)
    dur_dropoff1 = Dropout(dropout_rate)(dur_activ1)
    pitch2 = Dense(num_nodes)(pitch_dropoff1)
    pitch_bnorm2 = BatchNormalization()(pitch2)
    pitch_activ2 = ReLU()(pitch_bnorm2)
    pitch_dropoff2 = Dropout(dropout_rate)(pitch_activ2)
    duration2 = Dense(num_nodes)(dur_dropoff1)
    dur_bnorm2 = BatchNormalization()(duration2)
    dur_activ2 = ReLU()(dur_bnorm2)
    dur_dropoff2 = Dropout(dropout_rate)(dur_activ2)
    pitch = Dense(20 * 128)(pitch_dropoff2)
    duration = Dense(20 * 2)(dur_dropoff2)

    pitch_reshaped = Reshape((20, 128))(pitch)

    duration_reshaped = Reshape((20, 2))(duration)

    pitch_output = Softmax(axis=-1)(pitch_reshaped)

    duration_output = Dense(2, activation='relu', name='duration')(duration_reshaped)
    output_concat = concatenate([duration_output, pitch_output])
    generator = Model(inputs=random_inputs, outputs=output_concat)

    return generator


# Generator from v4 - most successful one to date
def define_generator2(latent_dim, dropout_rate, num_nodes):
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

# In[12]:


def generate_real_samples(dataset, n_samples, label_smoothing):
    """
    Makes a batch of real 20-note music samples
    :param dataset: the dataset containing 20-note music samples (x, 20, 130)
    :param n_samples: number of samples to generate
    :param label_smoothing: boolean, if true, real samples labeled as slightly less than 1 (0.9 for now)
    :return: training samples and target variable (set as 1 for real samples)
    """
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    if label_smoothing:
        # add stochastic noise
        y = np.ones((n_samples, 1)) - np.random.uniform(0.0, 0.1, (n_samples, 1))
    else:
        y = np.ones((n_samples, 1))
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
    # predict fake images
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


# # Model Training

# In[13]:


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    """
    training the GAN models
    :param g_model: generator keras model
    :param d_model: discriminator keras model
    :param gan_model: GAN model
    :param dataset: numpy array of real samples
    :param latent_dim: number of latent points for the random number input
    :param n_epochs: number of epochs to train
    :param n_batch: number of batches for each epoch
    :return: nothing
    """

    # generator model weights, sample output and loss history saved to timestamped folder
    # dataframe for recording history
    df_history = pd.DataFrame(columns=["epoch", "batch", "d_loss_real", "d_loss_fake", "g_loss"])
    df_history.to_csv(ts_str + '/loss_history.csv', index=False)
    # batches per epoch
    bat_per_epo = int(dataset.shape[0] / n_batch)
    real_batch = int(n_batch / 2)
    fake_batch = n_batch - real_batch
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get half_batch randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, real_batch, False)
            # update weights of discriminator model on real reamples
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' images
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, fake_batch)
            # update discriminator model weights on fake samples
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            for _ in range(5):
                # prepare points in latent space as input for the generator
                X_gan = generate_latent_points(latent_dim, n_batch)
                # label fake images as ones to train generator
                y_gan = np.ones((n_batch, 1))
                # update weights of the generator based on the discriminator's errors
                g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1= %.3f, d2= %.3f g= %.3f' %
                (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
            df_history.loc[len(df_history)] = [i+1, j+1, d_loss1, d_loss2, g_loss]
        # Generate a midi sample every 100 epochs
        if i % 100 == 0:
            pred = make_pred(g_model, latent_dim, 1)
            make_midi(pred, ts_str + "/ep_" + str(i) + ".mid")
            
            # append to history
            df_history.to_csv(ts_str + '/loss_history.csv', mode='a', index=False, header=False)
            df_history = pd.DataFrame(columns=["epoch", "batch", "d_loss_real", "d_loss_fake", "g_loss"])
            # save the model
            g_model.save(ts_str + '/gen_wgts_ep_' + str(i) + '.h5')
    # save the generator model
    g_model.save(ts_str + '/gen_wgts_ep_' + str(n_epochs) + '.h5')
    # save a final sample
    pred = make_pred(g_model, latent_dim, 1)
    make_midi(pred, ts_str + "/ep_" + str(n_epochs) + ".mid")
    # save loss history for analysis
    df_history.to_csv(ts_str + '/loss_history.csv', mode='a', index=False, header=False)


# In[17]:


# size of the latent space
latent_dim = 128
# number of nodes in dense layers
num_nodes = 256
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim, 0.2, num_nodes)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = np.load("training_samples/sample_2019-11-13-16-10.npy")
# Make timestamp
ts_str = time.strftime("%Y-%m-%d %H-%M", time.gmtime())
# create directory if not exist
if not os.path.exists(ts_str):
    os.makedirs(ts_str)


# In[18]:


# train model
train(generator, discriminator, gan_model, dataset, latent_dim, n_epochs=50, n_batch=600)


# In[ ]:




