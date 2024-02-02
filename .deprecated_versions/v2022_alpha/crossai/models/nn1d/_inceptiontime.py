from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Layer, Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Add
from ._layers import DropoutLayer


class InceptionTime(Model):
    def __init__(self, train_data_shape=None,
                 number_of_classes=1,
                 nb_filters=32, use_residual=True, use_bottleneck=True,
                 depth=6, kernel_size=41, bottleneck_size=32, drp_input=0,
                 drp_high=0,
                 kernel_initialize="he_uniform",
                 kernel_regularize=4e-5,
                 kernel_constraint=3):
        """

        Args:
            number_of_classes (int): The number of classes (>1). Default set to 1.
            train_data_shape (tuple (int,int)): The train data shape.
            nb_filters (int): The number of nb filters.
            use_residual (bool): Whether to use a residual block or not.
            use_bottleneck (bool): Whether to use a bottleneck layer or not.
            depth (int): The depth of the network.
            kernel_size (int): The kernel size of the network.
            bottleneck_size (int): The number of output filters in the convolution.
            drp_input (float): Range 0-1.
            drp_high (float): Range 0-1.
            kernel_initialize (str): The variance scaling initializer. Default: "he_uniform".
            kernel_regularize (float or str): Can be float or str in 1e-5 format.
            Regularizer to apply a penalty on the layer's kernel.
            kernel_constraint (int): The constraint of the value of the incoming weights. Default 3.

        Returns:

        """
        super(InceptionTime, self).__init__()
        self.number_of_classes = number_of_classes
        self.train_data_shape = train_data_shape
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size
        self.bottleneck_size = bottleneck_size
        self.drp_input = drp_input
        self.drp_high = drp_high
        self.spatial = False
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        if isinstance(self.kernel_regularize, str):
            self.kernel_regularize = float(kernel_regularize.replace("−", "-"))
        else:
            self.kernel_regularize = self.kernel_regularize
        self.kernel_constraint = kernel_constraint
        self.activation = "softmax"

        self.kernel_size = kernel_size - 1

        if self.kernel_regularize is not None:
            kernel_regularize = l2(self.kernel_regularize)
        if self.kernel_constraint is not None:
            kernel_constraint = MaxNorm(max_value=kernel_constraint, axis=[0, 1])

        # self.input_layer = Input(train_data_shape)

        self.drp0 = DropoutLayer(drp_rate=self.drp_input,
                                 spatial=False)
        self.inception_block = InceptionBlock(use_bottleneck=self.use_bottleneck,
                                              bottleneck_size=self.bottleneck_size, activation=self.activation,
                                              nb_filters=self.nb_filters, kernel_size=self.kernel_size,
                                              kernel_initialize=self.kernel_initialize,
                                              kernel_regularize=self.kernel_regularize,
                                              kernel_constraint=self.kernel_constraint
                                              )
        # print(self.inception_block.input)
        self.gap_layer = GlobalAveragePooling1D()

        # Dropout
        self.drp1 = DropoutLayer(drp_rate=self.drp_high, spatial=False)

        self.out = Dense(self.number_of_classes, activation=self.activation)

    def call(self, inputs):

        x = self.drp0(inputs)
        x = self.inception_block(x)
        x = self.gap_layer(x)
        x = self.drp1(x)
        x = self.out(x)

        return x

    def build_graph(self):
        x = Input(shape=self.train_data_shape)
        return Model(inputs=[x], outputs=self.call(x))


class InceptionBlock(Layer):
    def __init__(self, depth=6, use_bottleneck=True,
                 bottleneck_size=1, use_residual=True,
                 activation="softmax", nb_filters=32, kernel_size=41,
                 kernel_initialize="he_uniform", kernel_regularize=4e-5,
                 kernel_constraint=3):
        """

        Args:
            use_bottleneck:
            bottleneck_size:
            activation:
            nb_filters:
            kernel_size:
            kernel_initialize:
            kernel_regularize:
            kernel_constraint:
        """
        super(InceptionBlock, self).__init__()
        self.depth = depth
        self.use_bottleneck = use_bottleneck
        self.use_residual = use_residual
        self.bottleneck_size = bottleneck_size
        self.activation = activation
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        self.kernel_constraint = kernel_constraint
        self.inception_modules_list = list()
        self.residual_conv_list = list()
        self.bn_list = list()
        self.add_list = list()
        self.act_list = list()

        for d in range(self.depth):

            inception_module = InceptionModule(use_bottleneck=use_bottleneck, bottleneck_size=bottleneck_size,
                                               activation=activation, nb_filters=nb_filters, kernel_size=kernel_size,
                                               kernel_initialize=self.kernel_initialize,
                                               kernel_regularize=self.kernel_regularize,
                                               kernel_constraint=self.kernel_constraint)
            self.inception_modules_list.append(inception_module)

            if self.use_residual and d % 3 == 2:
                # print(inception_module.shape)
                residual_conv = Conv1D(filters=128, kernel_size=1,
                                       padding="same", use_bias=False,
                                       kernel_initializer=self.kernel_initialize,
                                       kernel_regularizer=self.kernel_regularize,
                                       kernel_constraint=kernel_constraint
                                       )
                self.residual_conv_list.append(residual_conv)
                self.bn_list.append(BatchNormalization())
                self.add_list.append(Add())
                self.act_list.append(Activation("relu"))

    def call(self, inputs):
        x = inputs
        inputs_res = inputs

        rc = 0
        for d in range(self.depth):
            x = self.inception_modules_list[d](x)
            if self.use_residual and d % 3 == 2:
                res_out = self.residual_conv_list[rc](inputs_res)
                shortcut_y = self.bn_list[rc](res_out)
                res_out = self.add_list[rc]([shortcut_y, x])
                x = self.act_list[rc](res_out)
                rc += 1
                inputs_res = x
        return x


class InceptionModule(Layer):
    """

    Args:
        input_tensor:
        use_bottleneck:
        bottleneck_size:
        activation:
        nb_filters:
        kernel_size:
        kernel_initialize:
        kernel_regularize:
        kernel_constraint:
        stride:

    Returns:

    """
    def __init__(self, use_bottleneck, bottleneck_size, activation, nb_filters, kernel_size,
                 kernel_initialize, kernel_regularize, kernel_constraint, stride=1, trainable=True):
        super(InceptionModule, self).__init__()
        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size
        self.activation = activation
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        self.kernel_constraint = kernel_constraint
        self.stride = stride
        self.input_inception = Conv1D(filters=bottleneck_size, kernel_size=1,
                                      padding="same", activation="linear",
                                      use_bias=False,
                                      kernel_initializer=self.kernel_initialize,
                                      kernel_regularizer=self.kernel_regularize,
                                      kernel_constraint=self.kernel_constraint
                                      )

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]
        print("kernel size list: ", kernel_size_s)

        self.conv_list = list()

        for i in range(len(kernel_size_s)):
            print("Inception filters: {} - kernel: {}".format(nb_filters, kernel_size_s[i]))
            self.conv_list.append(Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                         strides=self.stride, padding="same", activation=activation,
                                         use_bias=False,
                                         kernel_initializer=self.kernel_initialize,
                                         kernel_regularizer=self.kernel_regularize,
                                         kernel_constraint=self.kernel_constraint)
                                  )

        self.max_pool_1 = MaxPooling1D(pool_size=3, strides=self.stride, padding="same")

        self.conv_6 = Conv1D(filters=nb_filters, kernel_size=1,
                             padding="same", activation=activation, use_bias=False,
                             kernel_initializer=self.kernel_initialize,
                             kernel_regularizer=self.kernel_regularize,
                             kernel_constraint=self.kernel_constraint
                             )

        self.concat = Concatenate(axis=2)

    def call(self, inputs):
        if self.use_bottleneck and self.nb_filters > 1:
            x = self.input_inception(inputs)
        else:
            x = inputs
        convolutional_list = list()
        for conv in self.conv_list:
            convolutional_list.append(conv(x))

        x2 = self.max_pool_1(inputs)
        x2 = self.conv_6(x2)
        convolutional_list.append(x2)
        x_post = self.concat(convolutional_list)

        return x_post
