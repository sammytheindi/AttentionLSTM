""" TODO(sshah): Complete Module Docstring"""

# import unicodedata
# import re
# import numpy as np
# import os
# import time
# import shutil

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import Orthogonal
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras.layers import RNN, Dense, TimeDistributed, LSTMCell
from tensorflow.keras.activations import tanh, softmax


tf.random.set_seed(1)

class BahdanauAttention():
    pass

class LSTMAttentionCell(LSTMCell):
    """TODO(sshah): Complete Class Docstring"""
    # pylint: disable=too-many-instance-attributes
    # Justifiable number of attributes in this case
    def __init__(self, units, **kwargs):
        """TODO(sshah): Complete Function Docstring"""
        self._units = units
        self._attention_mode = False
        self._input_seq = None
        self._input_seq_shaped = None
        self._timesteps = None

        super(LSTMAttentionCell, self).__init__(units=self._units, **kwargs)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        """TODO(sshah): Complete Function Docstring"""
        # pylint: disable=attribute-defined-outside-init
        # Tensorflow documentation advises lazy initialization of weights
        self._memory_weight = TimeDistributed(
            Dense(self._units, name="memory_weight"))

        self._query_weight = Dense(self._units, name="query_weight")
        self._alignment_weight = Dense(1, name="alignment_weight")

        batch, input_dim = input_shape
        lstm_input = (batch, input_dim + input_dim)

        super(LSTMAttentionCell, self).build(lstm_input)

    @property
    def attention_mode(self):
        """TODO(sshah): Complete Function Docstring"""
        return self._attention_mode

    @attention_mode.setter
    def attention_mode(self, value):
        """TODO(sshah): Complete Function Docstring"""
        self._attention_mode = value

    @property
    def timesteps(self):
        """TODO(sshah): Complete Function Docstring"""
        return self._timesteps

    @property
    def input_seq(self):
        """TODO(sshah): Complete Function Docstring"""
        return self._input_seq

    def set_input_sequence(self, input_seq):
        """TODO(sshah): Complete Function Docstring"""
        self._input_seq = input_seq
        self._input_seq_shaped = self._memory_weight(self._input_seq)
        self._timesteps = tf.shape(self._input_seq)[-2]

    def call(self, inputs, states, constants):
        """TODO(sshah): Complete Function Docstring"""


        # hidden shape: (BATCH_SIZE, UNITS)
        _, hidden = states

        # hidden_repeated shape: (BATCH_SIZE, TIMESTEPS, UNITS)
        hidden_repeated = K.repeat(self._query_weight(hidden), self._timesteps)

        # alignment_score shape: (BATCH, TIMESTEPS, 1)
        alignment_score = self._alignment_weight(
            tanh(hidden_repeated + self._input_seq_shaped))

        # score_vector shape: (BATCH, TIMESTEPS, 1)
        score_vector = softmax(alignment_score, 1)

        # context_vector shape: (BATCH, INPUT_DIM)
        context_vector = K.sum(score_vector * self._input_seq, 1)

        # dec_inputs shape: (BATCH, INPUT_DIM + CONTEXT_DIM)
        dec_inputs = K.concatenate([inputs, context_vector], 1)

        
        res = super(LSTMAttentionCell, self).call(inputs=dec_inputs, states=states)
#         tf.print(res)

#         print(hidden_repeated.shape)
#         print(self.input_seq_shaped.shape)
#         print(self.input_seq_shaped.dtype)
# Note that in the addition case we are just adding the repeated hidden layer to the input state
#         print(tf.concat([hidden_repeated, self.input_seq_shaped], -1).shape)
        if self.attention_mode:
            print("Attention Mode On")
            print(K.reshape(score_vector, (-1, self._timesteps)).shape)
            # Reshapes to print attention matrix output
            return (K.reshape(score_vector, (-1, self._timesteps)), res[1])

        print("Attention Mode Off")
        print(res[0].shape)
        return res

class LSTMAttentionLayer(RNN):
    """TODO(sshah): Complete Class Docstring"""
    def __init__(self, cell, **kwargs):
        """TODO(sshah): Complete Function Docstring"""
        self._input_dim = None
        self._timesteps = None
        super(LSTMAttentionLayer, self).__init__(cell, **kwargs)

    @property
    def timesteps(self):
        """TODO(sshah): Complete Function Docstring"""
        return self._timesteps

    @property
    def input_dim(self):
        """TODO(sshah): Complete Function Docstring"""
        return self._input_dim

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        """TODO(sshah): Complete Function Docstring"""
#         print(input_shape)
        self._input_dim = input_shape[-1]
        self._timesteps = input_shape[-2]
        super(LSTMAttentionLayer, self).build(input_shape)

    def call(self, inputs, constants, **kwargs):
        """TODO(sshah): Complete Function Docstring"""
#         print(constants)
#         self.cell._dropout_mask = None
#         self.cell._recurrent_dropout_mask = None
        self.cell.set_input_sequence(constants[0])
        return super(LSTMAttentionLayer, self).call(inputs=inputs, constants=constants, **kwargs)

if __name__ == "__main__":
    INPUTS = tf.random.normal([1, 10, 2], dtype=tf.float32)
    MEMORY = tf.ones([1, 10, 2], dtype=tf.float32)

    ATTENTION_CELL = LSTMAttentionCell(units=20,
                                       recurrent_initializer=Orthogonal,
                                       kernel_initializer=Orthogonal)
    LAYER = LSTMAttentionLayer(ATTENTION_CELL, return_sequences=True, return_state=True)
    LAYER.cell.attention_mode = True

    OUTPUTS, FINAL_H, FINAL_C = LAYER(inputs=INPUTS, constants=MEMORY)
    print(OUTPUTS)
