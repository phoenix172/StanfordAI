import keras.metrics
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

def plot_loss_tf(history):
    fig,ax = plt.subplots(1,1, figsize = (4,3))
    ax.plot(history.history['loss'], label='loss')
    ax.set_ylim([0, 2])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('loss (cost)')
    ax.legend()
    ax.grid(True)
    plt.show()

def display_digit(X, Y):
    """ display a single digit. The input is one digit (400,). """
    fig = plt.figure(figsize=(7., 7.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(5, 5),  # creates 2x2 grid of axes
                     axes_pad=0.25,  # pad between axes in inch.
                     )
    images = random.sample(list(enumerate(X)), 25)
    for ax, im in zip(grid, images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im[1], cmap='gray')
        ax.set_title(Y[im[0]], fontsize=13)
        ax.set_axis_off()
    plt.show()

print(np.shape(X_train))
X_train_reshaped = X_train.reshape((60000,784))
print(np.shape(X_train_reshaped));

display_digit(X_train, Y_train)

tf.random.set_seed(1234)  # for consistent results
lambda_ = 0.01
model = Sequential(
    [
        ### START CODE HERE ###
        tf.keras.Input(shape=(784,)),
        tf.keras.layers.Dense(670, 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
        tf.keras.layers.Dense(248, 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
        tf.keras.layers.Dense(94, 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
        tf.keras.layers.Dense(30, 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
        tf.keras.layers.Dense(10, 'linear')

        ### END CODE HERE ###
    ], name="my_model"
)

model.summary()

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

history = model.fit(
    X_train_reshaped,Y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test.reshape((X_test.shape[0],784)),Y_test)
)
plot_loss_tf(history)

model.save("mnist-model.h5")

