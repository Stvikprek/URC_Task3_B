# -*- coding: utf-8 -*-
"""classifier.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1299BsrAkxM5mTeLzBHsgEnbXaJbJsJGj
"""

!pip install tensorflow[and-cuda]
!pip install opencv-python

# prompt: import opencv, keras and matplotlib

import cv2
from tensorflow.keras import datasets,layers,models
import matplotlib.pyplot as plt
import numpy as np

(training_images,training_labels),(testing_images,testing_labels) = datasets.cifar10.load_data()
training_images,testing_images = training_images/255,testing_images/255 # Normalize to be between 0 and 1
val_images,val_labels = testing_images[:1000],testing_labels[:1000]
testing_images,testing_labels = testing_images[1000:],testing_labels[1000:]

labels = ["Plane","Car","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck"]

model = models.Sequential() # Sequential, meaning we will be arranging our hidden layers in a particular sequence that will result in the final single-output. Good for singular input source

model.add(layers.RandomFlip('horizontal',input_shape = (32,32,3)))

model.add(layers.Conv2D(32,(3,3),activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation = 'relu'))

model.add(layers.Flatten())

model.add(layers.Dense(64,activation = 'relu'))
model.add(layers.Dense(10,activation = 'softmax'))

model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
model.summary()

shuff = np.random.permutation(len(training_images))
training_images = training_images[shuff]
training_labels = training_labels[shuff]

fitter = model.fit(training_images,training_labels,epochs = 15,validation_data = (val_images,val_labels))

loss,accuracy = model.evaluate(testing_images,testing_labels)
print(loss,accuracy)
model.save('75_model.model')

rand_image,rand_label = testing_images[np.random.randint(0,len(testing_images))],testing_labels[np.random.randint(0,len(testing_labels))]
plt.imshow(rand_image)
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.title(labels[model.predict(np.array([rand_image])).argmax()])
plt.show()