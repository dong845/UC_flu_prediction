from tensorflow_addons.layers import MultiHeadAttention
import keras
from keras.layers import Dense, Permute, GRU

class AttentionBlock(keras.Model):
    def __init__(self, name='AttentionBlock', num_heads=2, head_size=128, ff_dim=None, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)

        if ff_dim is None:
            ff_dim = head_size

        self.attention = MultiHeadAttention(num_heads=num_heads, head_size=head_size, dropout=dropout)
        self.attention_dropout = keras.layers.Dropout(dropout)
        self.attention_norm = keras.layers.LayerNormalization(epsilon=1e-6)

        self.ff_conv1 = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')
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


class ModelTrunk(keras.Model):
    def __init__(self, nodes, name='ModelTrunk', num_heads=2, head_size=128, ff_dim=159, num_layers=1, dropout=0.3, **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer1 = Dense(52, activation='relu')
        self.layer2 = Permute((2,1),input_shape=(159,52))
        self.layer3 = Dense(159, activation='relu')
        self.attention_layers = [AttentionBlock(num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)]
        self.gru = GRU(nodes, input_shape=(52, 159), dropout=0.3)
        self.layer4 = Dense(159)
        
    def call(self, inputs):
        input1, input2 = inputs
        x = self.layer1(input2)
        x = self.layer2(x)
        y = keras.layers.concatenate([input1, x], axis=2)
        y = self.layer3(y)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)
        x = self.gru(x)
        x = self.layer4(x)
        return x

