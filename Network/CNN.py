import tensorflow as tf

tf.get_logger().setLevel('ERROR')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from .CNN1 import CNN1
from .CNN2 import CNN2
from .CNN3 import CNN3
from .CNN4 import CNN4

class CNN(CNN1, CNN2, CNN3, CNN4):

    def __init__(self) -> None:
        pass
