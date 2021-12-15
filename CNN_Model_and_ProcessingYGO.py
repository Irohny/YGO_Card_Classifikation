import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Lambda
from skimage import filters, color, morphology, io
import skimage as si
import scipy.ndimage.morphology as scmorph
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import os

class CNN_Model():
    '''
    Class to create and train the CNN model
    '''
    def __init__(self):
        '''
        Constructor
        '''
        # empty property for the model
        self.model = []
        self.data_augmentation = []
        # defining default parameters
        # Input Parameter
        self.xPixel = 350
        self.yPixel = 350
        self.zLayer = 3
        # Learning Parameter
        self.loss_fn = tf.keras.losses.categorical_crossentropy
        self.acc_metric = tf.keras.metrics.CategoricalAccuracy()
        self.optimizer = tf.keras.optimizers.Adam()
        self.outLayer = 10
        self.split = 0.2
        self.BatchSize = 64
        self.reg = 0.01
        self.dropout = 0.5
        self.lr = 5e-4
        self.NEpoch = 200
        self.seed = 42
        self.NCluster = 10
        # Preprocessing Parameter
        self.sigma = 3
        self.it_Op = 1
        self.KernelOp = 9
        self.it_Dil = 3
        self.KernelDil = 3
        self.dist = 5
        # Augmentation Parameter
        self.Rotation = 0.1
        self.Zoom = 0.1
        
        # reset all seeds to 42 to reduce invariance between training runs
        os.environ['PYTHONHASHSEED']=str(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        # some line to correctly find some libraries in TF 2.1
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, enable=True)
        
    '''
        Convolutinal neral Network for Image Classification
    '''
    def build_CNN(self):
        
        # Defintion of DNN
        self.model = keras.Sequential([
                    layers.experimental.preprocessing.Rescaling(1, input_shape=(self.xPixel, self.yPixel, self.zLayer)),
            layers.Conv2D(16, 3, padding='same', kernel_regularizer=regularizers.l2(self.reg), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(self.reg), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(self.reg), activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(self.dropout),
            layers.Flatten(),
            layers.Dense(128, kernel_regularizer=regularizers.l2(self.reg), activation='relu'),
            layers.Dense(self.outLayer, activation='softmax', name='Output_Layer')
        ])
        
    
    def build_data_augmentation(self):
        # Data augmentation for robust training for small data sets
        self.data_augmentation = keras.Sequential([
                layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", input_shape=(self.xPixel, self.yPixel, self.zLayer)),
                layers.experimental.preprocessing.RandomRotation(self.Rotation),
                layers.experimental.preprocessing.RandomZoom(self.Zoom),
                #layers.experimental.preprocessing.Rescaling(1/255),
                #layers.experimental.preprocessing.Normalization(axis=1),
            
        ])
        #self.data_augmentation.compile(optimizer='adam',loss=self.loss_fn, metrics=self.acc_metric)
    '''
        Loading and Precrocessing Imge
        Input: path to Image
        Output: Foreground, Background, Unkown Areas
    ''' 
    def load_and_process(self, file, Training=True):
        img = io.imread(file)
        Nx, Ny, Nz = np.shape(img)
        # Hochkan,Seitkan Vergleich
        if (Nx < Ny):
            tmp = np.zeros((Ny, Nx, Nz))
            for ii in range(Nz):
                tmp[:,:,ii] = np.transpose(img[:,:,ii])
            img = tmp
        img = si.transform.resize(img, (self.xPixel, self.yPixel), anti_aliasing=True) #Resizing
        
        if Training==True:
            img = np.reshape(img[:,:,0:3], (1, self.xPixel, self.yPixel, self.zLayer))
            img = np.array(self.data_augmentation(img, training=False)).squeeze()
        
        img = filters.gaussian(img, sigma=self.sigma, multichannel=True) # Gaussian filter with radius 5
        img = color.rgb2gray(img)
        
        thresh = filters.threshold_otsu(img) # otsu algorithm treshold
        img = img > thresh
        img = np.abs(np.array(img, dtype='int')-1)
        for i in range(self.it_Op): # Opening
            img = morphology.area_opening(img, area_threshold=self.KernelOp)
        tmp = img
        for i in range(self.it_Dil): # Dilation
            tmp = morphology.dilation(tmp, morphology.square(self.KernelDil))
        BackGr = tmp
        distan = scmorph.distance_transform_edt(img, distances=self.dist)
        distan = distan > 0.7*distan.max()
        ForGr = np.abs(np.array(distan, dtype='int')-1)
        unkown = BackGr-ForGr
        
        out = np.zeros((self.xPixel, self.yPixel, self.zLayer))
        out[:,:,0] = ForGr
        out[:,:,1] = BackGr
        out[:,:,2] = unkown
        return out
  
    def load_and_process2(self, file, Training=True):
        img = io.imread(file)
        Nx, Ny, Nz = np.shape(img)
        # Hochkan,Seitkan Vergleich
        if (Nx < Ny):
            tmp = np.zeros((Ny, Nx, Nz))
            for ii in range(Nz):
                tmp[:,:,ii] = np.transpose(img[:,:,ii])
            img = tmp
        img = si.transform.resize(img, (self.xPixel, self.yPixel), anti_aliasing=True) #Resizing
        
        if Training==True:
            img = np.reshape(img[:,:,0:3], (1, self.xPixel, self.yPixel, self.zLayer))
            img = np.array(self.data_augmentation(img, training=False)).squeeze()
      
        proc = np.zeros((self.xPixel, self.yPixel, self.zLayer))
        for jj in range(self.zLayer):
             tmp = KMeans(n_clusters=self.NCluster, random_state=self.seed).fit(img[:, :, jj])
             proc[:, :, jj] = tmp.cluster_centers_[tmp.labels_]
             proc[:, :, jj] = proc[:, :, jj]/np.max(proc[:, :, jj]) 
        return proc      
    '''
        Create a Batch of Preprocessed Images
        Input: Path to Image
        Output: Batch of Preprocessed Images
    '''
    def create_data(self, data, Training=True):
        N = len(data)
        xBatch = np.zeros((N, self.xPixel, self.yPixel, self.zLayer))
        for idx in range(N):
            #xBatch[idx,:,:,:] = Lambda(self.load_and_process)(data[idx], Training)
            xBatch[idx,:,:,:] = Lambda(self.load_and_process2)(data[idx], Training)
        return xBatch

    '''
        Train the CNN Model for N Epochs
        Input: Data = Path to Image
               Label = Classlabel of Image
        Output: Best Trainings Accurancy
                Best CNN Model
    '''
    def train_model(self, Data, Label):
        # Create Model
        #Lambda(self.build_CNN)()
        tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.model.compile(optimizer='adam',loss=self.loss_fn, metrics=self.acc_metric)
        
        # Split Images into Test and Train Sets
        TrainImg, TestImg, TrainLabel, TestLabel = train_test_split(Data, Label, test_size=self.split, random_state=self.seed)
        Train_BatchPerEpoch = np.size(TrainLabel[:, 0])//self.BatchSize
        Test_BatchPerEpoch = np.size(TestLabel[:, 0])//self.BatchSize
        train_acc = np.zeros(self.NEpoch)
        train_loss = np.zeros(self.NEpoch)
        test_acc = np.zeros(self.NEpoch)
        
        # Train for N Epochs
        for epoch in range(self.NEpoch):
            # Training Loop
            print('Training Epoch %.f/%.f' % (epoch, self.NEpoch))
            for batch_idx in range(Train_BatchPerEpoch):
                print('Batch %.f/%.f' % (batch_idx+1, Train_BatchPerEpoch))
                # create Traingsbatch of Images and Labels
                start = int(batch_idx*self.BatchSize)
                end = int((batch_idx+1)*self.BatchSize)
                if end > len(TrainLabel):
                    end = len(TrainLabel)
                    
                xBatch = Lambda(self.create_data)(TrainImg[start:end])
                yBatch = TrainLabel[start:end]
                with tf.GradientTape() as tape:
                    y_pred = self.model(xBatch, training=True)
                    loss = self.loss_fn(yBatch, y_pred)
                   
                    gradients = tape.gradient(loss, self.model.trainable_weights)
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
                    self.acc_metric.update_state(yBatch, np.array(y_pred).squeeze())
            
            train_acc[epoch] = np.array(self.acc_metric.result())
            train_loss[epoch] = np.mean(loss)
            
            self.acc_metric.reset_states()
            
            print('Training Accuancy: %f' % (train_acc[epoch]))
        
            # Test Loop
            if Test_BatchPerEpoch == 0:
                Test_BatchPerEpoch = 1
            for batch_idx in range(Test_BatchPerEpoch):
                # create Testsets of images and Labels
                start = int(batch_idx*self.BatchSize)
                end = int((batch_idx+1)*self.BatchSize)
                if end > len(TestLabel):
                    end = len(TestLabel)
                
                xBatch = Lambda(self.create_data)(TestImg[start:end])
                yBatch = TestLabel[start:end]
                y_pred = self.model.predict_on_batch(xBatch)
                self.acc_metric.update_state(yBatch, np.array(y_pred).squeeze())
            test_acc[epoch] = np.array(self.acc_metric.result())
            
            self.acc_metric.reset_states()
            print('Test Accuancy: %f \n' % (test_acc[epoch]))
               
            if(test_acc[epoch] > 0.75):
                self.model.save('CNN_Model_Fluffal_ACC_%.2f.h5' %(int(epoch), np.round(test_acc, 2)))
        return train_acc, train_loss, test_acc 
    
    def predict(self, data):
        img = Lambda(self.create_data)(data, False)
        return np.array(self.model(img))
        
        