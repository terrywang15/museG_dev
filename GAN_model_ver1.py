#!/usr/bin/env python
# coding: utf-8

# # MScA Capstone
# # GAN Model for Generating Music Samples with Midi Data
# 
# ## Josh Goldberg, Rima Mittal, Terry Wang

# In[16]:


import time
import os
import numpy as np
from numpy.random import randn
from numpy.random import randint
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, concatenate, BatchNormalization, Input
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU, ReLU
from keras.layers import Dropout, Softmax
import pretty_midi


# # Model

# In[11]:


# Discriminator
def define_discriminator(in_shape=(20, 130)):
    model = Sequential()
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(256))
    model.add(Dropout(0.4))
    model.add(Dense(256))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


# Generator
def define_generator(latent_dim, dropout_rate, num_nodes):

    # Helper method (not inlined for clarity)
    def one_hot(x):
        return K.one_hot(K.argmax(x, axis=-1),
                          num_classes=latent_dim, axis=-1)

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
    pitch = Dense(20 * 128)(batchNorm3)
    duration = Dense(20 * 2)(batchNorm3)

    pitch_reshaped = Reshape((20, 128))(pitch)

    duration_reshaped = Reshape((20, 2))(duration)

    pitch_output = Softmax(axis=-1)(pitch_reshaped)

    duration_output = Dense(2, activation='relu', name='duration')(duration_reshaped)
    output_concat = concatenate([duration_output, pitch_output])
    generator = Model(inputs=random_inputs, outputs=output_concat)

    return generator


# Adversarial Model
def define_gan(generator, discriminator):
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


def generate_real_samples(dataset, n_samples):
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1)) # labeled as 1 
    return X, y


def generate_latent_points(latent_dim, n_samples):
    # 'n_samples' vectors of 'latent_dim' standard normal numbers each
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(generator, latent_dim, n_samples):
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

    gen_sample = generator.predict(generate_latent_points(latent_dim, num_samples))
    duration = gen_sample[:,:,:2]
    pitches = np.eye(128)[np.argmax(gen_sample[:,:,2:], axis=-1)]
    pred = np.concatenate((duration, pitches), axis=-1)

    return pred


def make_midi(pred, write_file_name):
    midi = pretty_midi.PrettyMIDI()
    notes = np.argmax(pred[0,:,2:], axis=-1)
    instr = pretty_midi.Instrument(0, name = 'Piano')
    for idx, note in enumerate(notes):
        note_to_add = pretty_midi.Note(velocity=100, pitch=int(note), start=min(pred[0,idx,0], pred[0,idx,1]), end=max(pred[0,idx,0], pred[0,idx,1]))
        instr.notes.append(note_to_add)
    midi.instruments.append(instr)
    midi.write(write_file_name)


# # Model Training

# In[13]:


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    # batches per epoch
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get half_batch randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update weights of discriminator model on real images
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' images
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights on fake images
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            for _ in range(2):
                # prepare points in latent space as input for the generator
                X_gan = generate_latent_points(latent_dim, n_batch)
                # label fake images as ones to train generator
                y_gan = np.ones((n_batch, 1))
                # update weights of the generator based on the discriminator's errors
                g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1= %.3f, d2= %.3f g= %.3f' %
                (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
        # Generate a midi sample every 100 epochs
        if i % 100 == 0:
            pred = make_pred(g_model, latent_dim, 1)
            make_midi(pred, ts_str + "/ep_" + str(i) + ".mid")
    # save the generator model
    g_model.save(ts_str + '/try_dc_generator.h5')


# In[17]:


# size of the latent space
latent_dim = 128
# length of noise vectors
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
train(generator, discriminator, gan_model, dataset, latent_dim, n_epochs=5000, n_batch=600)


# In[ ]:




