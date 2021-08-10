"""
@description: DynamicModel

@author: Zhipeng He
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input


def exp1_model(max_case_length, act_vocab_size, res_vocab_size, embed_dim = 36):
    act_inputs = layers.Input(shape=(max_case_length,))
    res_inputs = layers.Input(shape=(max_case_length,))

    x = layers.Embedding(input_dim=act_vocab_size, output_dim=embed_dim)(act_inputs)
    x = layers.LSTM(256)(x)

    y = layers.Embedding(input_dim=res_vocab_size, output_dim=embed_dim)(res_inputs)
    y = layers.LSTM(256)(y)

    out = layers.Concatenate()([x, y])

    outputs = layers.Dense(1, activation="sigmoid")(out)
    model_1 = tf.keras.Model(inputs=[act_inputs, res_inputs], outputs=outputs,
        name = "dynamic_model")
    return model_1
        


