""" TODO(sshah): Complete Module Docstring"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras.layers import RNN, LSTMCell
from attention_classes import MECH_AVAIL

class LSTMAttentionCell(LSTMCell):
    """Class for generating an LSTM Cell with Attention Capabilities.
    
    Args:
        units (int): The number of units in the hidden state of the LSTM Cell
        att_type (str): The attention mechanism to be used, from one of
            ['add', 'concat', 'content', 'location', 'general', 'dot', 'scaled-dot']
        attention_mode (Optional[bool]): return attention weights instead of hidden states. Defaults to False.
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments
    
    Attributes:
        _units (int): Number of units in hidden state of LSTM Cell
        _attention_mode (str): The attention mechanism to be used
        _timesteps (int): The number of timesteps in the decoder output. 
            Defaults to None but must be set in build method of the Recurrent Layer class.
        _context_dim (int): The dimensionality of the context vector. 
            Defaults to None but must be set in call method of the Cell class.
        _attention_layer (LuongAttentionBase): The attention layer. All classes derive from
            LuongAttentionBase class.
    """

    # pylint: disable=too-many-instance-attributes
    # Justifiable number of attributes in this case
    def __init__(self, units, att_type, attention_mode=False, **kwargs):
#         print("LSTMCell Init")
        self._units = units
        self._attention_mode = attention_mode
        self._timesteps = None
        self._context_dim = None
        
        assert att_type in MECH_AVAIL.keys(), 'Invalid attention mechanism, please choose from: {}'.format(MECH_AVAIL.keys())
        self._attention_layer = MECH_AVAIL[att_type](self._units)

        super(LSTMAttentionCell, self).__init__(units=self._units, **kwargs)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        """Build Method for LSTM Cell. 
        
        Will call superclass build method, initializing input dimensions of LSTM Cell to expect an
        input of dimension (input_dim + context_dim). This is due to the attention mechanism concatenating
        the input to the context vector as input to the decoder.
        
        Args:
            input_shape (tuple): shape of input to LSTM Cell, of the form (batch, input_dim)
        """

        # pylint: disable=attribute-defined-outside-init
        # Tensorflow documentation advises lazy initialization of weights

        batch, input_dim = input_shape
        lstm_input = (batch, input_dim + self._context_dim)

        super(LSTMAttentionCell, self).build(lstm_input)

    @property
    def context_dim(self):
        """int: get or set context vector dimension"""
        return self._context_dim

    @context_dim.setter
    def context_dim(self, value):
        self._context_dim = value

    # @property
    # def return_attention(self):
    #     """TODO(sshah): Complete Function Docstring"""
    #     return self._return_attention

    @property
    def attention_mode(self):
        """bool: get or set flag for lstm to output hidden states or attention weights"""
        return self._attention_mode

    @attention_mode.setter
    def attention_mode(self, value):
        self._attention_mode = value

    @property
    def timesteps(self):
        """int: get or set number of timesteps in encoder data"""
        return self._timesteps

    @timesteps.setter
    def timesteps(self, value):
        self._timesteps = value

    def call(self, inputs, states, constants):
        """Call Method for LSTM Cell
        
        Primary method used by the RNN layer, overrides superclass call method. Will lazily set timesteps 
        according to encoder outputs provided in constants. Will generate context vector depending on provided 
        attention mechanism and create inputs for superclass (decoder). Returns output of decoder.

        Args:
            inputs (Tensor): Inputs to the attention cell. These are also outputs of the encoder.
            states (Tuple): States from the previous 
            constants (Tuple):

        Returns:
            result (Tuple): LSTM internal state tuple of the form (state_h, [state_h, state_c]) if attention_mode enabled
            result(Tuple): Reshaped context vector and LSTM state tuple of the form (context, [state_h, state_c]) otherwise
        """  

        # hidden shape: (BATCH_SIZE, UNITS)
        _, hidden = states

        self._timesteps = constants[0].shape[-2]

        context_vector, score_vector = self._attention_layer(hidden, self._timesteps)

        # dec_inputs shape: (BATCH, INPUT_DIM + CONTEXT_DIM)
        dec_inputs = K.concatenate([inputs, context_vector], 1)

        result = super(LSTMAttentionCell, self).call(inputs=dec_inputs, states=states)

        if self.attention_mode:
            # Reshapes to print attention matrix output
            return (K.reshape(score_vector, (-1, self._timesteps)), result[1])
        return result

class LSTMAttentionLayer(RNN):
    """Class for creating Recurrent Layer with Attention Capabilities

    From Tensorflow 1.4+, keword arguments to the layer call method before and after model 
    compilation differed. 

    Before model compilation
        inputs (list): A list containing the tensor inputs, cell states and constants
        [inputs (Tensor), state_h (Tensor), state_c (Tensor), constants (Tensor)]

    After model compilation
        inputs (dict):

    The kwargs contain the training boolean
        {

        }

    After compilation, keyword arguments are passed as a dictionary of the form:
        {
            'initial_state': [state_h (Tensor), state_c (Tensor)]
            'constants': [constants (Tensor)]
            'training': training (bool)
        }

    """
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
        
        # Necessary when tensor graph is being constructed
        if len(kwargs) <= 1:
            input_fin = inputs
        # Necessary after compilation of tensor graph
        else:
#             print("inputs: {}".format(inputs))
#             print("initial_states: {}".format(kwargs['initial_state']))
#             print("constants: {}".format(kwargs['constants']))
            input_fin = [inputs] + kwargs['initial_state'] + kwargs['constants']
        
#         print("RNN Call Kwargs: {}".format(kwargs))
#         print("RNN Call Kwargs Length: {}".format(len(kwargs)))
        self.cell._attention_layer.set_input_sequence(input_fin[-1])
        return super(LSTMAttentionLayer, self).call(inputs, **kwargs)