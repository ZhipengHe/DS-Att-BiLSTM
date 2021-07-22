"""
@description: DynamicModel

@author: Zhipeng He
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DynamicModel(keras.Model):
    def __init__(self):
        # Initialize the necessary components of tf.keras.Model
        super(DynamicModel, self).__init__()
        # Now we initalize the needed layers - order does not matter.
        # -----------------------------------------------------------
        # Add initialization code here, including the layers that will be used in call(). e.g.,
        # layer1 = tf.keras.layers.BuiltInLayer(...)
        # layer2 = MyCustomLayer(...)

        # Flatten Layer
        self.flatten = layers.Flatten()
        # First Dense Layer
        self.dense1 = layers.Dense(128, activation = tf.nn.relu)
        # Output Layer
        self.dense2 = layers.Dense(10)

    # Forward pass of model - order does matter.
    def call(self, inputs):
        # Add the code for the model call here (process the input and return the output). e.g.,
        # x = layer1(input)
        # output = layer2(x)
        x = self.flatten(inputs)
        x = self.dense1(x)

        return self.dense2(x) # Return results of Output Layer

    # add your custom methods here
