from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np

DEFAULT_NUM_CLASSES = 30

def make_generator_model(seq_len, channels, noise_dim, num_classes=DEFAULT_NUM_CLASSES):
  """
  cGAN para WGAN (Generator)
  """
  noise_input = keras.Input(shape=(noise_dim,), name='noise_input')
  label_input = keras.Input(shape=(num_classes,), name='label_input')
  merged_input = layers.Concatenate()([noise_input, label_input])
  dense_size = seq_len // 8

  x = layers.Dense(dense_size * noise_dim, use_bias=False)(merged_input)
  x = layers.Reshape((dense_size, noise_dim))(x)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU()(x)
  x = layers.Conv1DTranspose(128, 5, strides=2, padding='same', use_bias=False)(x)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU()(x)
  x = layers.Conv1DTranspose(64, 5, strides=2, padding='same', use_bias=False)(x)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU()(x)
  output_seq = layers.Conv1DTranspose(
    channels, 5, strides=2, padding='same', use_bias=False, activation='softmax'
  )(x)

  model = Model(inputs=[noise_input, label_input], outputs=output_seq, name='C_Generator')
  return model

def make_critic_model(seq_len, channels, num_classes=DEFAULT_NUM_CLASSES):
  """
  cGAN para WGAN (Discriminator)
  """
  seq_input = keras.Input(shape=[seq_len, channels], name='seq_input')
  label_input = keras.Input(shape=(num_classes,), name='label_input')
  label_reshaped = layers.Dense(seq_len * channels // 4)(label_input)
  label_reshaped = layers.Reshape((seq_len, channels // 4))(label_reshaped)
  merged_input = layers.Concatenate(axis=-1)([seq_input, label_reshaped])

  x = layers.Conv1D(64, 5, strides=2, padding='same')(merged_input)
  x = layers.LeakyReLU()(x)
  x = layers.Conv1D(128, 5, strides=2, padding='same')(x)
  x = layers.LeakyReLU()(x)
  x = layers.Conv1D(256, 5, strides=2, padding='same')(x)
  x = layers.LeakyReLU()(x)
  x = layers.Flatten()(x)
  output_score = layers.Dense(1)(x)

  model = Model(inputs=[seq_input, label_input], outputs=output_score, name='C_Critic')
  return model