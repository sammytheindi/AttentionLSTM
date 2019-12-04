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
from tensorflow.keras.layers import RNN, Dense, TimeDistributed, LSTMCell, Input
from tensorflow.keras.activations import tanh, softmax

tf.random.set_seed(1)

class BahdanauAttention():
    pass

class CosineAttention():
    pass

class LocationBasedAttention():
    pass

class GeneralAttention():
    pass

class DotProductAttention():
    pass

class LSTMAttentionCell(LSTMCell):
    """TODO(sshah): Complete Class Docstring"""
    # pylint: disable=too-many-instance-attributes
    # Justifiable number of attributes in this case
    def __init__(self, units, attention_mode=False, **kwargs):
        print("LSTMCell Init")
        """TODO(sshah): Complete Function Docstring"""
        self._units = units
        self._attention_mode = attention_mode
        self._input_seq = None
        self._input_seq_shaped = None
        self._timesteps = None
        self._context_dim = None

        super(LSTMAttentionCell, self).__init__(units=self._units, **kwargs)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        print("LSTMCell Build")
        """TODO(sshah): Complete Function Docstring"""
        # pylint: disable=attribute-defined-outside-init
        # Tensorflow documentation advises lazy initialization of weights
        self._memory_weight = TimeDistributed(
            Dense(self._units, 
                  name="memory_weight"))

        self._query_weight = Dense(self._units, name="query_weight")
        self._alignment_weight = Dense(1, name="alignment_weight")

        batch, input_dim = input_shape
        lstm_input = (batch, input_dim + self._context_dim)

        super(LSTMAttentionCell, self).build(lstm_input)

    @property
    def context_dim(self):
        return self._context_dim

    @context_dim.setter
    def context_dim(self, value):
        self._context_dim = value

    @property
    def return_attention(self):
        """TODO(sshah): Complete Function Docstring"""
        return self._return_attention

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

    @timesteps.setter
    def timesteps(self, value):
        """TODO(sshah): Complete Function Docstring"""
        self._timesteps = value

    @property
    def input_seq(self):
        """TODO(sshah): Complete Function Docstring"""
        return self._input_seq

    def set_input_sequence(self, input_seq):
        """TODO(sshah): Complete Function Docstring"""
        print("Input_Seq: {}".format(input_seq))
        print(input_seq.shape[-2])
        self._input_seq = tf.reshape(input_seq, [-1, input_seq.shape[-2], input_seq.shape[-1]])
        print("Input_Seq_Reshaped: {}".format(self._input_seq))
        self._input_seq_shaped = self._memory_weight(self._input_seq)

    def call(self, inputs, states, constants):
        print("LSTMCell Call")
        print("inputs: {}".format(inputs))
        print("states: {}".format(states))
        print("constants: {}".format(constants))
        """TODO(sshah): Complete Function Docstring"""  
        
        # TODO(sshah): Implement Class based attention mechanism

        # hidden shape: (BATCH_SIZE, UNITS)
        _, hidden = states
        
        self._timesteps = constants[0].shape[-2]
        
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

        if self.attention_mode:
            # Reshapes to print attention matrix output
            return (K.reshape(score_vector, (-1, self._timesteps)), res[1])
        return res

class LSTMAttentionLayer(RNN):
    """TODO(sshah): Complete Class Docstring"""
    def __init__(self, cell, **kwargs):
        print("RNN Init")
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
        print("RNN Build")
        """TODO(sshah): Complete Function Docstring"""
        if isinstance(input_shape, list):
            self.cell.context_dim = input_shape[-1][-1]
            input_shape = input_shape[0]

        self._input_dim = input_shape[-1]
        self._timesteps = input_shape[-2]
        self.cell.timesteps = self._timesteps

        super(LSTMAttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        print("RNN Call")
        print("RNN Call Inputs: {}".format(inputs))
        """TODO(sshah): Complete Function Docstring"""
        # self.cell._dropout_mask = None
        # self.cell._recurrent_dropout_mask = None
        
        if len(kwargs) <= 1:
            input_fin = inputs
        else:
            print("inputs: {}".format(inputs))
            print("initial_states: {}".format(kwargs['initial_state']))
            print("constants: {}".format(kwargs['constants']))
            input_fin = [inputs] + kwargs['initial_state'] + kwargs['constants']
        
        print("RNN Call Kwargs: {}".format(kwargs))
        print("RNN Call Kwargs Length: {}".format(len(kwargs)))
        self.cell.set_input_sequence(input_fin[-1])
        return super(LSTMAttentionLayer, self).call(inputs, **kwargs)

if __name__ == "__main__":
    INPUTS = Input(shape=(None, 256))
    MEMORY = Input(shape=(11, 1024))

    ATTENTION_CELL = LSTMAttentionCell(units=units,
                                       attention_mode=True,
                                       recurrent_initializer=Orthogonal,
                                       kernel_initializer=Orthogonal)
    LAYER = LSTMAttentionLayer(ATTENTION_CELL, return_sequences=True, return_state=True)
    LAYER.cell.attention_mode = False
    OUTPUTS, FINAL_H, FINAL_C = LAYER(inputs=INPUTS, constants=MEMORY)
    print(OUTPUTS)