"""
Function to limit GPU memory growth to prevent crashing due to lack of memory.

Note: Code taken from https://www.tensorflow.org/guide/gpu?authuser=1#limiting_gpu_memory_growth
"""

import tensorflow as tf

def gpu_limiter():
    """
    Prevents maximisation of GPU memory growth to prevent overloading.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)