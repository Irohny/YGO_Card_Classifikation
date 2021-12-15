from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Lambda, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
import numpy as np

class Autoencoder(Model):
      def __init__(self):
            super(Autoencoder, self).__init__()
            self.latent_dim = 1000
            self.xDim = 340
            self.yDim = 400
            self.NChannel = 3
            # Encoder Model
            self.encoder = tf.keras.Sequential([
                        Input((self.xDim, self.yDim, self.NChannel)),
                        Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
                        Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
                        Flatten(),
                        Dense(self.latent_dim, activation='sigmoid')
                        ])
            self.Encoder_Layer = self.encoder.layers
            # Decoder Model      
            self.decoder = tf.keras.Sequential([
                        Input((self.latent_dim)),
                        Dense(self.Encoder_Layer[2].output_shape[1], activation='relu'),
                        Reshape(self.Encoder_Layer[1].output_shape[1:]),
                        Conv2DTranspose(8, 3, activation='relu', padding='same', strides=2),
                        Conv2DTranspose(16, 2, activation='relu', padding='same', strides=2),
                        Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')
                        ])
      def call(self, x):
            if len(np.shape(x))==3:
                  x = np.reshape(x, (1, self.xDim, self.yDim, self.NChannel))
                  
            encoded = self.encoder(x)
            return self.decoder(encoded)
            
      def predict(self, x):
            
            if len(np.shape(x))==3:
                  x = np.reshape(x, (1, self.xDim, self.yDim, self.NChannel))
                  
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return np.reshape(np.array(decoded), (self.xDim, self.yDim, self.NChannel))