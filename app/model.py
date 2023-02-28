'''The io module provides Python’s main facilities for dealing with various types of I/O. There are three main types of I/O: text I/O, binary I/O and raw I/O. These are generic categories, and various backing stores can be used for each of them. A concrete object belonging to any of these categories is called a file object. Other common terms are stream and file-like object.'''
import io
import numpy as np
import tensorflow as tf
'''Pillow is a Python library for working with images. It is a fork of the Python Imaging Library (PIL), which has not been updated since 2011. Pillow provides a simple and consistent interface for manipulating different image file formats, such as JPEG, PNG, BMP, and GIF.
Pillow makes it easy to perform common image processing tasks, such as resizing, cropping, rotating, and applying filters. It also supports more advanced features like color correction, blending, and compositing.'''
#pip install pillow
from PIL import Image
'''jsonify is a function in Flask that is used to convert a Python dictionary or list to a JSON string, and then wrap that string in a Flask Response object with the appropriate content-type header for JSON.
The jsonify function is commonly used in Flask applications when you want to return JSON data as a response to an HTTP request. It makes it easy to serialize Python data into a JSON format, which can then be sent to the client.'''
'''In Flask, request is a global object that represents the client's HTTP request that is being processed by the Flask application. It contains information about the request, such as the HTTP method, the URL path, the request headers, and the form data (if the request is a POST or PUT request).'''
from keras.utils import img_to_array

#Chargement du modèle amélioré
model = tf.keras.models.load_model('vgg16.h5')

#Fonction pour transformer la nouvelle image sur laquelle on fait la prédiction
# Preparing and pre-processing the image
def prepare_image(img):
    """
    prepares the image for the api call
    """
    img = Image.open(io.BytesIO(img)).convert('RGB')
    img = img.resize((150, 150))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img

#Fonction pour prédire un résultat sur cette image
def predict_result(img):
    """predicts the result"""
    return np.argmax(model.predict(img)[0])