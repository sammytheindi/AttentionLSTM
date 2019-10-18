""" TODO(sshah): Complete Module Docstring"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import Orthogonal
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras.layers import RNN, Dense, TimeDistributed, LSTMCell, Input, Embedding
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
        """TODO(sshah): Complete Function Docstring"""
        # pylint: disable=attribute-defined-outside-init
        # Tensorflow documentation advises lazy initialization of weights
        self._memory_weight = TimeDistributed(
            Dense(self._units, name="memory_weight"))

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
        self._input_seq = input_seq
        self._input_seq_shaped = self._memory_weight(self._input_seq)

    def call(self, inputs, states, constants):
        """TODO(sshah): Complete Function Docstring"""

        # TODO(sshah): Implement Class based attention mechanism

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

        print("dec_inputs")
        print(dec_inputs.shape)

        print("states")
        print(states[0].shape)

        res = super(LSTMAttentionCell, self).call(inputs=dec_inputs, states=states)

        if self.attention_mode:
            # Reshapes to print attention matrix output
            return (K.reshape(score_vector, (-1, self._timesteps)), res[1])
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
        if isinstance(input_shape, list):
            self.cell.context_dim = input_shape[-1][-1]
            input_shape = input_shape[0]

        self._input_dim = input_shape[-1]
        self._timesteps = input_shape[-2]
        self.cell.timesteps = self._timesteps

        super(LSTMAttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """TODO(sshah): Complete Function Docstring"""
        # self.cell._dropout_mask = None
        # self.cell._recurrent_dropout_mask = None
        self.cell.set_input_sequence(inputs[-1])
        return super(LSTMAttentionLayer, self).call(inputs, **kwargs)

if __name__ == "__main__":
    INPUTS = Input(shape=(10, 2))
    MEMORY = Input(shape=(10, 2))

    ATTENTION_CELL = LSTMAttentionCell(units=20,
                                       attention_mode=True,
                                       recurrent_initializer=Orthogonal,
                                       kernel_initializer=Orthogonal)
    LAYER = LSTMAttentionLayer(ATTENTION_CELL, return_sequences=True, return_state=True)
    LAYER.cell.attention_mode = False

    OUTPUTS, FINAL_H, FINAL_C = LAYER(inputs=INPUTS, constants=MEMORY)
    print(OUTPUTS)
