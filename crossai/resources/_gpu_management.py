import tensorflow as tf


def tf_exploit_physical_growth() -> None:
    """Sets memory growth for all PhysicalDevices available.

       This functionality prevents TensorFlow from allocating
       all memory on the device.
       Source: https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth

       Returns:

    """

    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
