import os
import tensorflow as tf
from crossai.models.nn2d import ResNet34, XceptionFunc


def load_model_oop():
    print("Load OOP model")

    num_classes = 2

    model_configuration = {"pooling_type": "avg",
                           "dense_layers": 2,
                           "dropout": False,
                           "dropout_first": False,
                           "dropout_rate": [0.5, 0.5],
                           "dense_units": [128, 128],
                           }

    model = ResNet34(input_shape=(224, 224, 3), num_classes=num_classes)
    model.build_graph()
    model.build(input_shape=(1, 224, 224, 3))
    print(model.summary())


def load_model_func():
    print("Load functional model.")

    shape = (299, 299, 3)
    model = XceptionFunc(input_shape=shape, num_classes=2).build_graph()
    #
    # model = Model()
    #
    # model.build()

    import os
    tf.keras.utils.plot_model(model=model, to_file=os.path.join(os.getcwd(), "xception_arch.png"))
    print(model.summary())


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
    # in case there is a problem with finding the libdevice during training export the XLA_FLAGS through python (DGX)
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda"

    load_model_func()
