from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout, MaxPooling1D)
from keras.layers.pooling import _Pooling1D
from keras.utils import conv_utils

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    rnn_layer = input_data
    for i in range(recur_layers):
        r1 = GRU(units, return_sequences=True)(rnn_layer)
#         (units, activation=activation, return_sequences=True, implementation=2, name='rnn')
        r1 = BatchNormalization()(r1)
        rnn_layer = r1

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(rnn_layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True), merge_mode='concat')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    # Add convolutional layer
    
#     filters = 400
    conv_border_mode = 'causal'
#     kernel_size = 11
    conv_1d = Conv1D(filters=200, kernel_size=21, #kernel=11 => 11*1/130 = 85ms; k=21 => 162ms
                     strides=1, 
                     dilation_rate=1,
                     padding=conv_border_mode,
#                      data_format='channels_last',
                     activation=None,
#                      use_bias=True,
                     name='conv1d')(input_data)
#     conv_1d = BatchNormalization(name='bn_conv_1d')(conv_1d)
#     conv_1d = Dropout(0.25)(conv_1d)
        
    maxpool_1d = MaxPooling1D_U(pool_size=3, 
                              strides=2, 
                              padding='same', 
                              data_format='channels_last',
                              name='maxpool1d')(conv_1d)
    activ_r1 = Activation('relu', name="relu_conv1")(maxpool_1d)
    bn_1 = BatchNormalization(name='bn_1')(activ_r1)
    bn_1 = Dropout(0.4)(bn_1)    
    
    conv_1d_2 = Conv1D(filters=200, kernel_size=11, 
                     strides=1, 
                     dilation_rate=1,
                     padding=conv_border_mode,
#                      activation='relu',
                     name='conv1d_2')(bn_1)
    
#     conv_1d_2 = MaxPooling1D_U(pool_size=3, 
#                               strides=2, 
#                               padding='same', 
#                               data_format='channels_first',
#                               name='maxpool_2')(conv_1d_2)
    conv_1d_2 = Activation('relu', name="relu_conv2")(conv_1d_2)
    conv_1d_2 = BatchNormalization(name='bn_2')(conv_1d_2)

    conv_1d_2 = Dropout(0.4)(conv_1d_2)
#     conv_1d_3 = Conv1D(filters, kernel_size, 
#                      strides=1, 
#                      dilation_rate=4,
#                      padding=conv_border_mode,
# #                      activation='relu',
#                      name='conv1d_3')(conv_1d_2)
#     conv_1d_4 = Conv1D(filters, kernel_size, 
#                      strides=1, 
#                      dilation_rate=8,
#                      padding=conv_border_mode,
#                      activation='relu',
#                      name='conv1d_4')(conv_1d_2)
#     # Add batch normalization
#     conv_1d_4 = BatchNormalization(name='bn_conv_1d_4')(conv_1d_4)
#     conv_1d_4 = Dropout(0.4)(conv_1d_4)
    # Add a recurrent layer
    rnn = Bidirectional(GRU(200, dropout=0.4, return_sequences=True, name='rnn'), merge_mode='concat')(conv_1d_2)
    rnn = BatchNormalization()(rnn)
    
    rnn2 = Bidirectional(GRU(200, dropout=0.4, return_sequences=True, name='rnn2'), merge_mode='concat')(rnn)
    rnn2 = BatchNormalization()(rnn2)
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(rnn2)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
#     d1 = lambda x, dil: cnn_output_length(
#         x, kernel_size, conv_border_mode, 1, dilation=dil)
    
    model.output_length = lambda x: x/2
    print(model.summary())
    return model

class MaxPooling1D_U(_Pooling1D):
    """Reimplementing MaxPool due to missing data format parameter in version 2.0.9 of Keras
    """

    def __init__(self, pool_size=2, strides=None,
                 padding='valid', data_format='channels_last', **kwargs):
        super(MaxPooling1D_U, self).__init__(pool_size, strides,
                                           padding,
                                           **kwargs)
        self.data_format = data_format

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        output = K.pool2d(inputs, pool_size, strides,
                          padding, data_format, pool_mode='max')
        return output
    
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            steps = input_shape[2]
            features = input_shape[1]
        else:
            steps = input_shape[1]
            features = input_shape[2]
        length = conv_utils.conv_output_length(steps,
                                               self.pool_size[0],
                                               self.padding,
                                               self.strides[0])
        if self.data_format == 'channels_first':
            return (input_shape[0], features, length)
        else:
            return (input_shape[0], length, features)

    def call(self, inputs):
        dummy_axis = 2 if self.data_format == 'channels_last' else 3
        inputs = K.expand_dims(inputs, dummy_axis)   # add dummy last dimension
        output = self._pooling_function(inputs=inputs,
                                        pool_size=self.pool_size + (1,),
                                        strides=self.strides + (1,),
                                        padding=self.padding,
                                        data_format=self.data_format)
        return K.squeeze(output, dummy_axis)  # remove dummy last dimension