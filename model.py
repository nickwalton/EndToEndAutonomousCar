import csv
import cv2
import numpy as np
import sklearn
import matplotlib.image as mpimg
from PIL import Image
import pdb
from random import randint
from scipy.misc import imresize

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.layers.core import Dropout
from sklearn.model_selection import train_test_split
from keras.models import load_model


#Hyperparameters
load_data = True
use_side_cameras = True
model_name = 'jungle_model.h5'

def train_model(data_path, restart_model=True,epochs=3):
    if(load_data):
        X_train = np.load(data_path+'X_train.npy')
        y_train = np.load(data_path+'y_train.npy')
    else:

        lines = []
        with open(data_path+'driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
                
        images = []
        angles = []
        for line in lines:
            image = mpimg.imread(line[0])
            
            # Reduce image size to allow room in memory (Also high pixel resolution not needed)
            image = imresize(image,(64,64))
            angle = float(line[3])
            
            #Remove some straight image and angles
            rand_remove = randint(1,10)
            if(abs(angle) < 0.05 and rand_remove > 3):
                continue
            
            #Take off top 24 pixels
            images.append(image[20:64])
            angles.append(angle)
            
            # Augment with flipped Images
            flipped_image = cv2.flip(image,1)
            flipped_angle = -1.0 * angle
            images.append(flipped_image[20:64])
            angles.append(flipped_angle)

	        # Use left and right cameras and slightly adjust 
            if(use_side_cameras):
                left_image = mpimg.imread(line[1])
                right_image = mpimg.imread(line[2])
                left_angle = angle + 1.2
                right_angle = angle - 1.2

                images.append(imresize(left_image,(64,64))[20:64])
                images.append(imresize(right_image,(64,64))[20:64])
                angles.append(left_angle)
                angles.append(right_angle)

        X_train = np.array(images)
        y_train = np.array(angles)

        np.save(data_path+'X_train',X_train)
        np.save(data_path+'y_train',y_train)

    input_shape = (44,64,3)

    print(X_train.shape)

    if(restart_model):
        model = Sequential()
        model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
        model.add(Conv2D(16, (4,4),strides=(1,1),activation="relu"))
        model.add(MaxPooling2D())
        model.add(Conv2D(24, (3,3),strides=(1,1),activation="relu"))
        model.add(MaxPooling2D())
        model.add(Conv2D(36, (3,3),strides=(1,1),activation="relu"))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(124))
        model.add(Dense(48))
        model.add(Dense(1))
        
        model.compile(loss='mse',optimizer='adam')
        
    else:
        model = load_model(model_name)

    model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=epochs)
    model.save(model_name)


main_data_path = 'data/new_data/'
side_data_path = 'data/data2/'
jungle_data_path = 'data/jungle_data/'
#train_model(main_data_path, restart_model=True,epochs=3)
#train_model(side_data_path, restart_model=False,epochs=3)
train_model(jungle_data_path, restart_model=True,epochs=2)








