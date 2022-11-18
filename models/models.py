from tensorflow_addons.layers import MultiHeadAttention
import keras
from keras.layers import Dense, Permute, GRU
from keras.layers.core import Flatten

class AttentionBlock(keras.Model):
    def __init__(self, name='AttentionBlock', num_heads=2, head_size=128, ff_dim=None, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)

        if ff_dim is None:
            ff_dim = head_size

        self.attention = MultiHeadAttention(
            num_heads=num_heads, head_size=head_size, dropout=dropout)
        self.attention_dropout = keras.layers.Dropout(dropout)
        self.attention_norm = keras.layers.LayerNormalization(epsilon=1e-6)

        self.ff_conv1 = keras.layers.Conv1D(
            filters=ff_dim, kernel_size=1, activation='relu')
        self.ff_dropout = keras.layers.Dropout(dropout)
        self.ff_norm = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.attention([inputs, inputs])
        x = self.attention_dropout(x)
        x = self.attention_norm(inputs + x)

        x = self.ff_conv1(x)
        x = self.ff_dropout(x)
        x = self.ff_norm(inputs + x)
        return x
    
class AttentionBlock_simple(keras.Model):
    def __init__(self, name='AttentionBlock_simple', num_heads=2, head_size=128, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)

        self.attention = MultiHeadAttention(
            num_heads=num_heads, head_size=head_size, dropout=dropout)
        self.attention_dropout = keras.layers.Dropout(dropout)
        self.attention_norm = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.attention([inputs, inputs])
        x = self.attention_dropout(x)
        x = self.attention_norm(x)
        return x
    
class Conv1dBlock(keras.Model):
    def __init__(self, name='Conv1dBlock', ff_dim=None, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)

        self.ff_conv1 = keras.layers.Conv1D(filters=ff_dim, kernel_size=3, strides=2, padding="same", activation='relu')
        self.ff_dropout1 = keras.layers.Dropout(dropout)
        self.ff_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ff_conv2 = keras.layers.Conv1D(filters=ff_dim, kernel_size=3, strides=2, padding="same", activation='relu')
        self.ff_dropout2 = keras.layers.Dropout(dropout)
        self.ff_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.ff_conv1(inputs)
        x = self.ff_dropout1(x)
        x = self.ff_norm1(x)
        x = self.ff_conv2(x)
        x = self.ff_dropout2(x)
        x = self.ff_norm2(x)
        return x


class Model_v1(keras.Model):
    def __init__(self, nodes, name='Model_v1', num_heads=2, head_size=128, ff_dim=169, num_layers=1, dropout=0.3, **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer1 = Permute((2, 1), input_shape=(60, 169))
        self.conv1ds =  Conv1dBlock(ff_dim=169)
        self.gru = GRU(nodes, input_shape=(169, 15), dropout=0.3)
        self.layer4 = Dense(159)

    def call(self, x):
        x = self.layer1(x)
        x = self.conv1ds(x)
        x = self.gru(x)
        x = self.layer4(x)
        return x


class Model_v2(keras.Model):
    def __init__(self, nodes, name='Model_v2', num_heads=2, head_size=128, ff_dim=159, num_layers=1, dropout=0.3, **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer1 = Dense(52, activation='relu')
        self.layer2 = Permute((2, 1), input_shape=(159, 52))
        self.layer3 = Dense(159, activation='relu')
        self.attention_layers = [AttentionBlock(
            num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)]
        self.gru = GRU(nodes, input_shape=(52, 159), dropout=0.3)
        self.layer4 = Dense(159)

    def call(self, inputs):
        input1, input2 = inputs
        x = self.layer1(input2)
        x = self.layer2(x)
        x = keras.layers.concatenate([input1, x], axis=2)
        x = self.layer3(x)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)
        x = self.gru(x)
        x = self.layer4(x)
        return x
    
class Model_v21(keras.Model):
    def __init__(self, nodes, name='Model_v21', num_heads=2, head_size=64, dropout=0.3, **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer1 = Permute((2, 1), input_shape=(52, 159))
        
        self.attention_layers = AttentionBlock_simple(num_heads=num_heads, head_size=head_size, dropout=dropout)
        self.conv1d_layers = Conv1dBlock(ff_dim=159)
        self.layer2 = Dense(32, activation='relu')
        self.layer3 = Dense(64, activation='relu')
        
        self.layer5 = Flatten()
        
        self.gru = GRU(nodes, input_shape=(159, 32), dropout=0.3)
        
        self.layer4 = Dense(159)

    def call(self, inputs):
        input1, input2 = inputs
        # x1 = self.attention_layers(input2)   # 271, 159, 8
        x1 = self.layer5(input2)
        x1 = self.layer3(x1)
        
        x2 = self.layer1(input1)   # 271, 159, 52
        x2 = self.conv1d_layers(x2)  # 271, 159, 13
        x2 = self.layer2(x2)
        x2 = self.gru(x2)
        print("x2:", x2.shape)
        
        x = keras.layers.concatenate([x1, x2], axis=-1)
        x = self.layer4(x)
        return x
    
