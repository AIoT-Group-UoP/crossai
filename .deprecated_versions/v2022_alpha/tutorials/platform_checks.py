import platform
import sys
import pandas as pd
import sklearn as sk
import tensorflow as tf


def check_platform():
    print("System:", platform.system())
    print("System Release:", platform.release())
    print("System Version:", platform.version())
    print("Python Version:", platform.python_version())


def check_libraries_versions():
    print("Python Platform:", platform.platform())
    print("TensorFlow Version:", tf.__version__)
    print("Keras Version:", tf.keras.__version__)
    print()
    print("Python:", sys.version)
    print("Pandas:", pd.__version__)
    print("Scikit-Learn:", sk.__version__)
    gpu = len(tf.config.list_physical_devices('GPU')) > 0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")


if __name__ == '__main__':
    check_platform()
    check_libraries_versions()
