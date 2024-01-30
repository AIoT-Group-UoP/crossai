import tensorflow as tf
from ._layers import DenseBlock, DenseDropBlock


class VGG16(tf.keras.Model):

    def __init__(
        self,
        *,
        input_shape: tuple,
        num_classes: int,
        pooling_type: str,
        dense_layers: int,
        dense_units: list,
        dropout: bool,
        dropout_first: bool,
        dropout_rate: list,
    ) -> None:

        super().__init__()

        self.bottom_arch = tf.keras.applications.vgg16.VGG16(
            include_top=False,
            weights=None,
            input_shape=input_shape,
            pooling=pooling_type,
        )

        self._input_shape = input_shape
        if dropout:
            self.dense_block = DenseDropBlock(dense_layers, dense_units=dense_units, drop_rate=dropout_rate,  activation="relu", drop_first=dropout_first)
        else:
            self.dense_block = DenseBlock(dense_layers, dense_units=dense_units, activation="relu")

        self.out = tf.keras.layers.Dense(num_classes, activation='softmax')


    def call(self, x):
        x = self.bottom_arch(x)
        x = self.dense_block(x)
        return self.out(x)
    

    def build_graph(self):
        x = tf.keras.Input(shape=self._input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))