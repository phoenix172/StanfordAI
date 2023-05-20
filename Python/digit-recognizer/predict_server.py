import os

import base64
import io
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

from klein import Klein
from twisted.web import server, static
from twisted.internet import reactor

import tensorflow as tf
import numpy as np

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

model = tf.keras.models.load_model("mnist-model.h5")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
app = Klein()
# staticFileDir = os.environ['HOME']
staticFileDir = './webroot'

@app.route('/', branch=True)
def staticFiles(request):
    return static.File(staticFileDir)

@app.route('/predict', methods=['POST'])
def predict(request):
    image = request.content.read().partition(b'base64,')[2]
    dec_image = base64.b64decode(image)
    grayscale_image = Image.open(io.BytesIO(dec_image)).convert('L')

    plt.imshow(grayscale_image, interpolation='nearest')
    plt.show()

    imgArray = tf.keras.utils.img_to_array(grayscale_image)
    resized = tf.image.resize(imgArray, (28,28), method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=True, antialias=False)
    reshaped = resized.reshape(1,28*28)

    prediction = np.argmax(tf.nn.softmax(model.predict(reshaped)))

    return str(prediction)


app.run('0.0.0.0', 9000)