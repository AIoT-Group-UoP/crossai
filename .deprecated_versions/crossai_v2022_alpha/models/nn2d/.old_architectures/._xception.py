import tensorflow as tf
from crossai.models.nn2d._layers import DenseBlock, DenseDropBlock


class Xception(tf.keras.Model):

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
        kernel_regularize: float,
        kernel_constraint: float,
    ) -> None:

        super().__init__()
        self.pooling_type = pooling_type

        self.bottom_arch = tf.keras.applications.xception.Xception(
            include_top=False,
            weights=None,
            input_shape=input_shape,
            pooling=pooling_type,
        )

        for layer in self.bottom_arch.layers:
            if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                print(layer.name)
                # layer.add_loss(lambda layer=layer: tf.keras.regularizers.l2(0.00001)(layer.kernel))
                layer.add_loss(lambda layer=layer: tf.keras.constraints.MaxNorm(3)(layer.kernel))

        self._input_shape = input_shape
        if dropout:
            self.dense_block = DenseDropBlock(dense_layers, dense_units=dense_units, drop_rate=dropout_rate,  activation="relu", drop_first=dropout_first)
        else:
            self.dense_block = DenseBlock(dense_layers, dense_units=dense_units, activation="relu")

        self.out = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.bottom_arch(x)
        if self.pooling_type is None:
            x = tf.keras.layers.Flatten()(x)
        x = self.dense_block(x)
        return self.out(x)

    def build_graph(self):
        x = tf.keras.Input(shape=self._input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
