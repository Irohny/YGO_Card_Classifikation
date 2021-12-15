from autoencoder import Autoencoder
from reading_class import read_images
from tensorflow.keras import losses
import os
from glob import glob
#from sys import intern
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import numpy as np

'''
      hyperparameter
'''
NEpochs = 50
dimVec = np.linspace(300, 10000, 97)

path = './Fluffal_Deck'
Itype = '.JPG'

files = []
#start_dir = os.getcwd()
pattern   = "*.JPG"

for dir,_,_ in os.walk(path):
    files.extend(glob(os.path.join(dir,pattern))) 
    
reader = read_images()


img_tensor = reader.read_list(files)
train_set, test_set = train_test_split(img_tensor, test_size=0.2)
del img_tensor

for ii in dimVec:
      print('Train LD Size: %f' %(ii))
      model = Autoencoder()
      model.latent_dim = ii
      model.compile(optimizer='adam', loss=losses.MeanSquaredError())
      a = model.fit(train_set, train_set, validation_data=(test_set, test_set), epochs=NEpochs)
      model.encoder.save('./Model/Encoder_%sLD_%f.h5' % (ii, np.round(np.min(a.history['val_loss']),6)))
      model.decoder.save('./Model/Decoder_%sLD_%f.h5' % (ii, np.round(np.min(a.history['val_loss']),6)))
      
      fig, axs = plt.subplots(1,4, figsize=(28,7))
      axs[0].plot(a.history['loss'], 'r', label='Train Loss')
      axs[0].plot(a.history['val_loss'], 'b', label='Test Loss')
      axs[0].grid('on')
      axs[0].set_ylabel('MSE')
      axs[0].set_xlabel('Epoch')
      axs[0].legend()
      axs[0].set_title('LD %s Min Val Loss %s' % (ii, np.round(np.min(a.history['val_loss']),6)))
      axs[0].set_ylim([0, 0.05])
      for jj in range(3):
            z = random.randint(0, len(files)-1)
            img = reader.reading(files[z])
            axs[jj+1].imshow(reader.inv_norm(np.reshape(model(img), (340, 400, 3))))
      fig.savefig('./Model/Model_LD_%f_%f.png' % (ii, np.round(np.min(a.history['val_loss']),6)))
      
      del model, fig, axs, a 
      