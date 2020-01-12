""" TODO(sshah): Complete Module Docstring"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras.layers import RNN, LSTMCell
from attention_classes import MECH_AVAIL

# Input
#Dense, TimeDistributed, activation tanh
# tf.random.set_seed(1)

class LSTMAttentionCell(LSTMCell):
    """TODO(sshah): Complete Class Docstring"""
    # pylint: disable=too-many-instance-attributes
    # Justifiable number of attributes in this case
    def __init__(self, units, att_type, attention_mode=False, **kwargs):
#         print("LSTMCell Init")
        """TODO(sshah): Complete Function Docstring"""
        self._units = units
        self._attention_mode = attention_mode
        self._timesteps = None
        self._context_dim = None
        
        assert att_type in MECH_AVAIL.keys(), 'Invalid attention mechanism, please choose from: {}'.format(MECH_AVAIL.keys())
        self.attention_layer = MECH_AVAIL[att_type](self._units)

        super(LSTMAttentionCell, self).__init__(units=self._units, **kwargs)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
#         print("LSTMCell Build")
        """TODO(sshah): Complete Function Docstring"""
        # pylint: disable=attribute-defined-outside-init
        # Tensorflow documentation advises lazy initialization of weights

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

    def call(self, inputs, states, constants):
#         print("LSTMCell Call")
#         print("inputs: {}".format(inputs))
#         print("states: {}".format(states))
#         print("constants: {}".format(constants))
        """TODO(sshah): Complete Function Docstring"""  
        
        # TODO(sshah): Implement Class based attention mechanism

        # hidden shape: (BATCH_SIZE, UNITS)
        _, hidden = states
        
        self._timesteps = constants[0].shape[-2]
        
        context_vector = self.attention_layer(hidden, self._timesteps)
#         print('context vector: {}'.format(context_vector))

        # dec_inputs shape: (BATCH, INPUT_DIM + CONTEXT_DIM)
        dec_inputs = K.concatenate([inputs, context_vector], 1)
#         print('dec_inputs: {}'.format(dec_inputs))

        res = super(LSTMAttentionCell, self).call(inputs=dec_inputs, states=states)

        if self.attention_mode:
            # Reshapes to print attention matrix output
            return (K.reshape(score_vector, (-1, self._timesteps)), res[1])
        return res

class LSTMAttentionLayer(RNN):
    """TODO(sshah): Complete Class Docstring"""
    def __init__(self, cell, **kwargs):
#         print("RNN Init")
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
#         print("RNN Build")
        """TODO(sshah): Complete Function Docstring"""
        if isinstance(input_shape, list):
            self.cell.context_dim = input_shape[-1][-1]
            input_shape = input_shape[0]

        self._input_dim = input_shape[-1]
        self._timesteps = input_shape[-2]
        self.cell.timesteps = self._timesteps

        super(LSTMAttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
#         print("RNN Call")
#         print("RNN Call Inputs: {}".format(inputs))
        """TODO(sshah): Complete Function Docstring"""
        # self.cell._dropout_mask = None
        # self.cell._recurrent_dropout_mask = None
        
        if len(kwargs) <= 1:
            input_fin = inputs
        else:
#             print("inputs: {}".format(inputs))
#             print("initial_states: {}".format(kwargs['initial_state']))
#             print("constants: {}".format(kwargs['constants']))
            input_fin = [inputs] + kwargs['initial_state'] + kwargs['constants']
        
#         print("RNN Call Kwargs: {}".format(kwargs))
#         print("RNN Call Kwargs Length: {}".format(len(kwargs)))
        self.cell.attention_layer.set_input_sequence(input_fin[-1])
        return super(LSTMAttentionLayer, self).call(inputs, **kwargs)

