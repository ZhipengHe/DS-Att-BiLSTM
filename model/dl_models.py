"""
@description: DynamicModel

@author: Zhipeng He
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input

class D_Model(Model):
    """
    """

    def __init__(self):
        # Initialize the necessary components of tf.keras.Model
        super(D_Model, self).__init__()
        # Now we initalize the needed layers - order does not matter.

        # Embedding layer for activity
        self.act_embedding = layers.experimental.preprocessing.TextVectorization(output_mode='binary')
        # Embedding layer for resource group
        self.res_embedding = layers.experimental.preprocessing.TextVectorization(output_mode='binary')

        # activity lstm layer
        self.act_lstm = layers.LSTM(10)
        # resource group lstm layer
        self.res_lstm = layers.LSTM(10)
        # elapsed time lstm layer
        self.t_lstm = layers.LSTM(10)

        # concatenate layer
        self.concatenate = layers.Concatenate()

        # dropout layer
        self.dropout = layers.Dropout(0.2)

        # output layer for binary classification
        self.outlayer = layers.Dense(1, activation='sigmoid')

    # Forward pass of model - order does matter.
    def call(self, inputs):
        # Add the code for the model call here (process the input and return the output). e.g.,
        # x = layer1(input)
        # output = layer2(x)

        # activity feature
        act = act_embedding(inputs[0])
        act = act_lstm(act)

        res = res_embedding(inputs[1])
        res = res_lstm(res)

        t = t_lstm(inputs[2])

        output = concatenate(act, res, t)
        output = outlayer(output)

        # Return results of Output Layer
        return output

    # add your custom methods here
    def model(self):
        x = Input(shape=(24, 24, 3)) # change shape here
        return Model(inputs=[x], outputs=self.call(x))

        


