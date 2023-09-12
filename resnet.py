from tensorflow.keras.layers import Conv2D,Dense,LayerNormalization,ReLU
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import tensorflow as tf

class data_augmentation(tf.keras.layers.Layer):
    def __init__(self):
      super().__init__()
      self.data_augmentation = Sequential(
        [
          layers.RandomFlip("horizontal"),
          layers.RandomRotation(0.1),
          layers.RandomZoom(0.1)
        ])

    def call(self,x):
      return self.data_augmentation(x)
    

class Residual_main(tf.keras.layers.Layer):
    def __init__(self,filter,kernel_size):
        super().__init__()
        self.seq = Sequential([
            Conv2D(filter,kernel_size,padding='same'),
            LayerNormalization(),
            ReLU(),
            Conv2D(filter,kernel_size,padding='same'),
            # LayerNormalization(),
            # ReLU(),
            # Conv2D(filter*2,kernel_size,padding='same'),
        ])

    def call(self,x):
        return self.seq(x)
    

class project(tf.keras.layers.Layer):
    def __init__(self,units):
        super().__init__()
        self.seq = Sequential([
            Dense(units),
            LayerNormalization()
        ])

    def call(self,x):
        return self.seq(x)
    

def add_residual_block(input,filter,kernel_size):
    out = Residual_main(filter,kernel_size)(input)
    res = input

    if out.shape[-1] != input.shape[-1]:
        res = project(out.shape[-1])(input)

    return layers.add([res,out])
