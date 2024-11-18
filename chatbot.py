# to ignore the warnings
import warnings
warnings.filterwarnings ("ignore")

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
import tensorflow

from google.colab import drive
drive.mount('/content/drive')
#The code above is used to mount a Google Drive folder in a Google Colab session.

train_datagen = image.ImageDataGenerator(rescale=1/255, horizontal_flip=True,zoom_range=0.2)
train_data = train_datagen.flow_from_directory(directory='/content/drive/MyDrive/Dataset/Train',
target_size= (256,256), batch_size=16, class_mode='binary')

from google.colab import drive
drive.mount('/content/drive')

train_data.class_indices

val_datagen = image.ImageDataGenerator (rescale=1/255)
val_data= val_datagen. flow_from_directory (directory='/content/drive/MyDrive/Dataset/Val',
target_size = (256,256), batch_size=16, class_mode='binary')

model=Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(256,256,3)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.25))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit_generator(train_data, steps_per_epoch=8, epochs=10, validation_data = val_data)

# model = load_model("covid.h5")

from tensorflow.keras.models import load_model
model.save("covid.h5")
model.save(r"/content/drive/MyDrive/Dataset/covid.h5")

# model_1 = load_model("/content/covid.h5")
from tensorflow.keras.models import load_model

model = load_model("/content/drive/MyDrive/Dataset/covid.h5")

path='/content/drive/MyDrive/Dataset/Val/Covid/covid-19-pneumonia-14-PA.png'
img=tensorflow.keras.utils.load_img(path, target_size=(256,256,3))
img=img_to_array(img)/255
plt.imshow(img)
img=np.array([img])
img.shape



model.predict(img)
