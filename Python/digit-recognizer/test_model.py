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
def display_digit(X, Y, model):
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
        actual = Y[im[0]]
        reshaped = im[1].reshape((1,28*28))
        prediction = np.argmax(tf.nn.softmax(model.predict(reshaped)))
        ax.set_title(str(actual) + " P:" + str(prediction), fontsize=13)
        ax.set_axis_off()
    plt.show()

print(np.shape(X_train))
X_train_reshaped = X_train.reshape((60000,784))
print(np.shape(X_train_reshaped));

#display_digit(X_train, Y_train)

model = tf.keras.models.load_model("mnist-model.keras")
model.summary()
display_digit(X_test, Y_test, model)

print(X_test.shape[0])

result = model.evaluate(X_test.reshape((X_test.shape[0], 784)), Y_test)
print(result)



