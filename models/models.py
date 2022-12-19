import keras
from keras.layers import Dense, Permute, GRU
from keras.layers.core import Flatten
    
class Conv1dBlock(keras.Model):
    def __init__(self, name='Conv1dBlock', ff_dim=None, strides=1, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)

        self.ff_conv1 = keras.layers.Conv1D(filters=ff_dim, kernel_size=3, strides=strides, padding="same", activation='relu')
        self.ff_dropout1 = keras.layers.Dropout(dropout)
        self.ff_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ff_conv2 = keras.layers.Conv1D(filters=ff_dim, kernel_size=3, strides=strides, padding="same", activation='relu')
        self.ff_dropout2 = keras.layers.Dropout(dropout)
        self.ff_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ff_conv3 = keras.layers.Conv1D(filters=ff_dim, kernel_size=3, strides=1, padding="same", activation='relu')
        self.ff_dropout3 = keras.layers.Dropout(dropout)
        self.ff_norm3 = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x1 = self.ff_conv1(inputs)
        #x1 = self.ff_dropout1(x1)
        x1 = x1+inputs
        # x = self.ff_norm1(x)
        x2 = self.ff_conv2(x1)
        #x2 = self.ff_dropout2(x2)
        x2 = x2 + x1
        # x = self.ff_norm2(x)
        x3 = self.ff_conv3(x2)
        x3 = x3 + x2
        # x = self.ff_dropout3(x)
        # x = self.ff_norm3(x)
        return x3

# conv1d
class Model_v1(keras.Model):
    def __init__(self, nodes, name='Model_v1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer1 = Dense(169, activation='relu')
        self.conv1ds =  Conv1dBlock(ff_dim=169)
        self.gru = GRU(nodes, input_shape=(60, 169), dropout=0.3)
        self.layer4 = Dense(64, activation='relu')
        # self.layer5 = Dense(64, activation='relu')
        self.layer6 = Dense(159)

    def call(self, x):
        x = self.layer1(x)
        x = self.conv1ds(x)
        x = self.gru(x)
        x = self.layer4(x)
        # x = self.layer5(x)
        x = self.layer6(x)
        return x

# conv1d + combine 1
class Model_v11(keras.Model):
    def __init__(self, nodes, name='Model_v11', **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer1 = Dense(159, activation='relu')
        self.conv1ds =  Conv1dBlock(ff_dim=159)
        self.gru = GRU(nodes, input_shape=(60, 159), dropout=0.3)
        self.layer4 = Dense(64, activation='relu')
        # self.layer5 = Dense(64, activation='relu')
        self.layer6 = Dense(159)

    def call(self, x):
        x = self.layer1(x)
        x = self.conv1ds(x)
        x = self.gru(x)
        x = self.layer4(x)
        # x = self.layer5(x)
        x = self.layer6(x)
        return x

# conv1d + combine 2
class Model_v21(keras.Model):
    def __init__(self, nodes, name='Model_v21', **kwargs):
        super().__init__(name=name, **kwargs)        
        self.layer1 = Dense(159, activation='relu')
        self.layer2 = Dense(52, activation='relu')
        self.layer3 = Permute((2, 1), input_shape=(8, 52))
        
        self.conv1d_layers = Conv1dBlock(ff_dim=167, strides=1)
        self.layer4 = Dense(159, activation='relu')
        self.gru = GRU(nodes, input_shape=(52, 159), dropout=0.3)
        self.layer5 = Dense(64, activation='relu')
        # self.layer6 = Dense(64, activation='relu')
        self.layer7 = Dense(159)

    def call(self, inputs):
        input1, input2 = inputs
        x = self.layer1(input1)
        
        x1 = self.layer2(input2)
        x1 = self.layer3(x1)
        y = keras.layers.concatenate([x, x1], axis=-1)
        y = self.conv1d_layers(y)
        y = self.layer4(y)
        y = self.gru(y)
        y = self.layer5(y)
        # y = self.layer6(y)
        y = self.layer7(y)
        return y

# conv1d + combine 3  
class Model_v31(keras.Model):
    def __init__(self, nodes, name='Model_v31', **kwargs):
        super().__init__(name=name, **kwargs)        
        self.layer1 = Dense(159, activation='relu')
        self.layer2 = Dense(52, activation='relu')
        self.layer3 = Permute((2, 1), input_shape=(159, 52))
        
        self.conv1d_layers = Conv1dBlock(ff_dim=159, strides=1)
        self.layer4 = Dense(159, activation='relu')
        self.gru = GRU(nodes, input_shape=(52, 159), dropout=0.3)
        self.layer5 = Dense(64, activation='relu')
        # self.layer6 = Dense(64, activation='relu')
        self.layer7 = Dense(159)

    def call(self, inputs):
        input1, input2 = inputs
        x = self.layer1(input1)
        
        x1 = self.layer2(input2)
        x1 = self.layer3(x1)
        y = x*0.7+x1*0.3
        y = self.conv1d_layers(y)
        y = self.layer4(y)
        y = self.gru(y)
        y = self.layer5(y)
        # y = self.layer6(y)
        y = self.layer7(y)
        return y