import unittest

import tensorflow as tf

from attention_classes import AdditiveAttention
from lstm_attention import LSTMAttentionCell, LSTMAttentionLayer
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import Input

class MyTestCase(unittest.TestCase):
    def setUp(self):
        print("Setting Up")
        self.setUpTestCases()

class test_mechanisms(MyTestCase):
    def setUpTestCases(self):
        print("Hello")
        tf.random.set_seed(1)
        
        self.units=3
        self.attention_mode=False
        self.att_type = None

        self.inputs = Input(shape=(None, 2))
        self.memory = Input(shape=(1, self.units))
        

    def test_attention_failure(self):
        self.att_type = 'add'
        try:
            outputs = self.mechanism_script()
        except ExceptionType:
            self.fail("myFunc() raised ExceptionType unexpectedly!")
            
    def test_additive_attention(self):
        self.att_type = 'add'
        outputs = self.mechanism_script()
        self.assertEqual([None, None, self.units], outputs.shape.as_list())
        
    def test_concat_attention(self):
        self.att_type = 'concat'
        outputs = self.mechanism_script()
        self.assertEqual([None, None, self.units], outputs.shape.as_list())
        
    def test_content_attention(self):
        self.att_type = 'content'
        outputs = self.mechanism_script()
        self.assertEqual([None, None, self.units], outputs.shape.as_list())
        
    def test_location_based_attention(self):
        self.att_type = 'location'
        outputs = self.mechanism_script()
        self.assertEqual([None, None, self.units], outputs.shape.as_list())
        
    def test_general_attention(self):
        self.att_type = 'general'
        outputs = self.mechanism_script()
        self.assertEqual([None, None, self.units], outputs.shape.as_list())
        
    def test_dot_product_attention(self):
        self.att_type = 'dot'
        outputs = self.mechanism_script()
        self.assertEqual([None, None, self.units], outputs.shape.as_list())
    
    def test_scaled_dot_product_attention(self):
        self.att_type = 'scaled-dot'
        outputs = self.mechanism_script()
        self.assertEqual([None, None, self.units], outputs.shape.as_list())
            
    def mechanism_script(self):
        attention_cell = LSTMAttentionCell(units=self.units,
                                           att_type=self.att_type,
                                           attention_mode=self.attention_mode,
                                           recurrent_initializer=Orthogonal,
                                           kernel_initializer=Orthogonal)
        layer = LSTMAttentionLayer(attention_cell, return_sequences=True, return_state=True)
        layer.cell.attention_mode = False
        outputs, final_h, final_c = layer(inputs=self.inputs, constants=self.memory)
        return outputs
    
if __name__ == '__main__':
    unittest.main()
