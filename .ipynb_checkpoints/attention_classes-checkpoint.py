from abc import ABC,abstractmethod
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, TimeDistributed
from tensorflow.keras.activations import tanh, softmax

class LuongAttentionBase(ABC, tf.keras.layers.Layer):
    def __init__(self, units):
        super(LuongAttentionBase, self).__init__()
        self._units = units
        self._input_seq = None
        self._input_seq_shaped = None
        
    def call(self, hidden, timesteps):
        hidden_transformed = self.transform_hidden(hidden)
        hidden_repeated = K.repeat(hidden_transformed, timesteps)
        input_seq_transformed = self._input_seq_shaped
        alignment_score = self.calculate_alignment(hidden_repeated, input_seq_transformed)
        score_vector = softmax(alignment_score, 1)
        context_vector = K.sum(score_vector * self.input_seq, 1)
        
        return context_vector
    
    def set_input_sequence(self, input_seq):
        """TODO(sshah): Complete Function Docstring"""
#         print("Input_Seq: {}".format(input_seq))
#         print(input_seq.shape[-2])
        self._input_seq = tf.reshape(input_seq, [-1, input_seq.shape[-2], input_seq.shape[-1]])
#         print("Input_Seq_Reshaped: {}".format(self._input_seq))
        self._input_seq_shaped = self.shape_input_seq(self._input_seq)
        
    @abstractmethod
    def shape_input_seq(self, input_seq):
        pass
    
    @abstractmethod
    def transform_hidden(self, hidden):
        pass
    
    @abstractmethod
    def calculate_alignment(self, hidden, input_seq):
        pass
    
    @property
    def input_seq(self):
        """TODO(sshah): Complete Function Docstring"""
        return self._input_seq
    
class AdditiveAttention(LuongAttentionBase):
    def __init__(self, units):
        super(AdditiveAttention, self).__init__(units)
        self._memory_weight = TimeDistributed(
            Dense(self._units, 
                  name="memory_weight"))
        
        self._query_weight = Dense(self._units, name="query_weight")
        self._alignment_weight = Dense(1, name="alignment_weight")
    
    def transform_hidden(self, hidden):
        return self._query_weight(hidden)
    
    def calculate_alignment(self, hidden_repeated, input_seq_transformed):
        return self._alignment_weight(
            tanh(hidden_repeated + input_seq_transformed))
    
    def shape_input_seq(self, input_seq):
        return self._memory_weight(input_seq)
    
class ConcatAttention(LuongAttentionBase):
    def __init__(self, units):
        super(ConcatAttention, self).__init__(units)
        self._query_weight = Dense(self._units, name="query_weight")
        self._alignment_weight = Dense(1, name="alignment_weight")
        
    def transform_hidden(self, hidden):
        return hidden
    
    def calculate_alignment(self, hidden_repeated, input_seq_transformed):
        return self._alignment_weight(
            tanh(self._query_weight(K.concatenate([hidden_repeated, self._input_seq], -1))))
    
    def shape_input_seq(self, input_seq):
        return input_seq
        
class ContentBasedAttention(LuongAttentionBase):
    def __init__(self, units):
        super(ContentBasedAttention, self).__init__(units)
    
    def transform_hidden(self, hidden):
        return hidden
    
    def calculate_alignment(self, hidden_repeated, input_seq_transformed):
        cos_sim_numerator = tf.reduce_sum(tf.multiply(hidden_repeated, input_seq_transformed), -1, keepdims=True)
        mag_hidden = tf.linalg.norm(hidden_repeated, axis=-1, keepdims=True) 
        mag_input = tf.linalg.norm(input_seq_transformed, axis=-1, keepdims=True)
        cos_sim_denominator = tf.math.multiply(mag_hidden, mag_input)
        return tf.math.divide(cos_sim_numerator, cos_sim_denominator)
    
    def shape_input_seq(self, input_seq):
        return input_seq

class LocationBasedAttention(LuongAttentionBase):
    def __init__(self, units):
        super(LocationBasedAttention, self).__init__(units)
        self._alignment_weight = Dense(1, name="alignment_weight")
        
    def transform_hidden(self, hidden):
        return hidden
    
    def shape_input_seq(self, input_seq):
        return input_seq
    
    def calculate_alignment(self, hidden_repeated, input_seq_transformed):
        return self._alignment_weight(hidden_repeated)

class GeneralAttention(LuongAttentionBase):
    def __init__(self, units):
        super(GeneralAttention, self).__init__(units)
        self._memory_weight = TimeDistributed(
            Dense(self._units, 
                  name="memory_weight"))
        
    def transform_hidden(self, hidden):
        return hidden
    
    def shape_input_seq(self, input_seq):
        return self._memory_weight(input_seq)
    
    def calculate_alignment(self, hidden_repeated, input_seq_transformed):
        return tf.reduce_sum(tf.multiply(hidden_repeated, input_seq_transformed), -1, keepdims=True)
    
class DotProductAttention(LuongAttentionBase):
    def __init__(self, units):
        super(DotProductAttention, self).__init__(units)
        
    def transform_hidden(self, hidden):
        return hidden
    
    def shape_input_seq(self, input_seq):
        return input_seq
    
    def calculate_alignment(self, hidden_repeated, input_seq_transformed): 
        return tf.reduce_sum(tf.multiply(hidden_repeated, input_seq_transformed), -1, keepdims=True)

class ScaledDotProductAttention(DotProductAttention):
    def __init__(self, units):
        super(ScaledDotProductAttention, self).__init__(units)
        
    def calculate_alignment(self, hidden_repeated, input_seq_transformed):
        sqrt_d = tf.sqrt(tf.cast(tf.shape(hidden_repeated)[-1], tf.float32))
        return tf.divide(tf.reduce_sum(tf.multiply(hidden_repeated, input_seq_transformed), -1, keepdims=True), sqrt_d)

MECH_AVAIL = {
            'add': AdditiveAttention, 
            'concat': ConcatAttention, 
            'content': ContentBasedAttention, 
            'location': LocationBasedAttention, 
            'general': GeneralAttention, 
            'dot': DotProductAttention, 
            'scaled-dot': ScaledDotProductAttention
            }