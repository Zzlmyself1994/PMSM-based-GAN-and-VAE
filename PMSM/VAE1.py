'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras import objectives
from keras.datasets import mnist

np.random.seed(1111)  # for reproducibility

batch_size = 5
n = 7
m = 2
hidden_dim = 256
epochs = 50
epsilon_std = 1.0
use_loss = 'xent' # 'mse' 均方误差 or 'xent'交叉熵  or  'mae'绝对误差

decay = 1e-4 # weight decay, a.k. l2 regularization
use_bias = True

## Encoder
x = Input(batch_shape=(batch_size, n))
h_encoded = Dense( units=hidden_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias, activation='tanh')(x)
z_mean = Dense(units=m, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias)(h_encoded)
z_log_var = Dense(units=m, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias)(h_encoded)


## Sampler
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal_variable(shape=(batch_size, m), mean=0.,
                                       scale=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(m,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(hidden_dim , kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias, activation='tanh')
decoder_mean = Dense(n, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias, activation='sigmoid')

## Decoder
h_decoded = decoder_h(z)
x_hat = decoder_mean(h_decoded)


## loss
def vae_loss(x, x_hat):
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    xent_loss = n * objectives.binary_crossentropy(x, x_hat)
    mse_loss = n * objectives.mse(x, x_hat)
    mae_loss = n * objectives.mae(x, x_hat)
    if use_loss == 'xent':
        return xent_loss + kl_loss
    elif use_loss == 'mse':
        return mse_loss + kl_loss
    elif use_loss == 'mae':
        return  mae_loss + kl_loss
    else:
        raise Exception('Nonknow loss!')

vae = Model(x, x_hat)
vae.compile(optimizer='rmsprop', loss=vae_loss)

# 读取数据及标签
training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename='PMSM_train1.csv',
      target_dtype=np.int,
      features_dtype=np.float64)

test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename='PMSM_test1.csv',
    target_dtype=np.int,
    features_dtype=np.float64)


vae.fit(x=training_set.data, y=training_set.target,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(test_set.data, test_set.target))




#print(x)
##----------Visualization----------##
# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(test_set.data, batch_size=batch_size)
fig = plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=test_set.target)
plt.colorbar()
plt.show()
fig.savefig('z.pmsm_{}.png'.format(use_loss))


# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(m,))
_h_decoded = decoder_h(decoder_input)
_x_hat = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_hat)

n= 30
x_gen= np.zeros((n, n))
grid_x1 = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y1 = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x1):
    for j , xi in enumerate(grid_y1):
        z_sample= np.array([[xi, yi]])
        x_decoded= generator.predict(z_sample)
        print(j, x_decoded)


'''
###################### display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
################### linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
##################### to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

fig = plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
fig.savefig('x_{}.png'.format(use_loss))

######################## data imputation
figure = np.zeros((digit_size * 3, digit_size * n))
x = test_set.data[:batch_size,:]
x_corupted = np.copy(x)
x_corupted[:, 300:400] = 0
x_encoded = vae.predict(x_corupted, batch_size=batch_size).reshape((-1, digit_size, digit_size))
x = x.reshape((-1, digit_size, digit_size))
x_corupted = x_corupted.reshape((-1, digit_size, digit_size))
for i in range(n):
    xi = x[i]
    xi_c = x_corupted[i]
    xi_e = x_encoded[i]
    figure[:digit_size, i * digit_size:(i+1)*digit_size] = xi
    figure[digit_size:2 * digit_size, i * digit_size:(i+1)*digit_size] = xi_c
    figure[2 * digit_size:, i * digit_size:(i+1)*digit_size] = xi_e

fig = plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
fig.savefig('i_{}.png'.format(use_loss))

'''