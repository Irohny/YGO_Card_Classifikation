import numpy as np
from CNN_Model import CNN_Model
import matplotlib.pyplot as plt
import os
from glob import glob
from sys import intern

NxPixel = 300
NyPixel = 200

path = './Fluffal_Deck'
Itype = '.JPG'

files = []
#start_dir = os.getcwd()
pattern   = "*.JPG"

for dir,_,_ in os.walk(path):
    files.extend(glob(os.path.join(dir,pattern))) 
    
Classes = [x[0] for x in os.walk(path)]
NData = len(files)
NClasses = len(Classes)-1
Label = np.zeros((NData, NClasses))

for jj in range(NData):
    for ii in range(1, NClasses+1):
        N = len(Classes[ii])
        if intern(files[jj][0:N]) is intern(Classes[ii]):
            Label[jj, ii-1] = 1

Model = CNN_Model()
Model.xPixel  = 400
Model.yPixel = 230
Model.outLayer = NClasses

Model.build_CNN()
Model.build_data_augmentation()
TrainAcc, TestAcc, TrainLoss, TestLoss = Model.train_model(files, Label)

plt.plot(TrainAcc, 'r', label='Train')
plt.plot(TestAcc, 'b', label='Test')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

N = len(files)
mat = np.zeros((NClasses, NClasses))
for ii in range(N):
    y = np.argmax(Model.predict([files[ii]]))
    x = np.argmax(Label[ii])
    mat[x,y] = mat[x,y]+1
    
for jj in range(NClasses):
      mat[jj, :] = mat[jj, :]/np.sum(mat[jj, :])
      
np.save('Confusionmatrix.npy', mat)
Erg = {'TrainAcc':TrainAcc, 'TestAcc':TestAcc, 'TrainLoss':TrainLoss, 'TestLoss':TestLoss}
np.save('Model_Fitting.npy', Erg)

plt.matshow(mat)
plt.ylabel('Prediction')
plt.xlabel('Label')