## Setup
- Make sure you have tensorflow,numpy,matplotlib installed
- The dataset used for training is the standard CIFAR10 image dataset in which images are grouped under 10 classifications. This was chosen for ease of access. The dataset contains several RGB 32X32 images for training and testing purposes

## Script 
- First we load the dataset into several arrays, two of which (`training_images` and `testing_images`) contain the pixel by pixel image data and the other two containing respective labels. The image data is reduced to having values from 0 to 1 instead of a 0 to 255 as this will lead to better optimization. The testing dataset is also divided into validation and actual testing sets
```
(training_images,training_labels),(testing_images,testing_labels) = datasets.cifar10.load_data()
training_images,testing_images = training_images/255,testing_images/255 # Normalize to be between 0 and 1
val_images,val_labels = testing_images[:1000],testing_labels[:1000]
testing_images,testing_labels = testing_images[1000:],testing_labels[1000:]
```

- If you know the labels of the dataset, you are also free to make a `labels` list, otherwise you can let the labels remain as is

### The model and training
- The model used is a Sequential CNN (Sequential meaning the input data is processed by layers arranged in a set sequence). A CNN(Convolutional Neural Network)uses Convolution, a mathematical process of multiplying the elements of sequences/matrices and summing the products in a certain way to produce another sequence/matrix, to extract meaningful information from the image and detect features
```
model = models.Sequential()
```

- The first layer is a data augmentation layer that randomly flips our input RGB 32X32 image laterally. This helps us provide more varied data to the model during training, especially in consequent epochs (epochs: iterations over the provided training set)
```
model.add(layers.RandomFlip('horizontal',input_shape = (32,32,3)))
```

- The next few layers perform Convolutions using a 3X3 kernel (first one has 32 filters while the rest have 64) with the activation function being `f(x) = max(0,x)`(relu). The output of convolution is down sampled via MaxPooling using a kernel of 2X2 (4 pixels are represented by one, which has the maximum value out of the four)
```
model.add(layers.Conv2D(32,(3,3),activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
```

- The next layer flattens the 3 channel image data (3 matrices representing RGB values) into a uni-dimensional array to be fed into the "dense" part of the neural network. This part contains of 64 neurons fully connected with our output layer of 10 neurons
```
model.add(layers.Flatten())

model.add(layers.Dense(64,activation = 'relu'))
model.add(layers.Dense(10,activation = 'softmax'))
```

- Next step, we set our optimizer ('adam' is an optimizer for gradient descent), loss function(sparse categorical cross entropy as our labels are pure integers) and a metric to optimize (accuracy in this case). `model.summary()` can be used to get a summary of your model architecture
```
model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
model.summary()
```

- Now we train the model over our training set using the `model.fit()` method. In this case, the model is trained over the entire training set to prevent underfitting, while our randomflip layer helps keep data varied and prevent overfitting
```
model.fit(training_images,training_labels,epochs = 15,validation_data = (val_images,val_labels))
```
- You can also save the model using `model.save('filename.model)` or `model.save(filename.h5)`

### Using the saved model
- The model can be loaded using the `tensorflow.keras.models.load_model()` function 
- If loading in the image using OpenCV, make sure to convert it to be an RGB 32X32 image (that gets later turned into a numpy array)
```
img = cv2.imread('horse.jpg',cv2.IMREAD_COLOR)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(32,32))

prediciton = model.predict(np.array([img])) # The model expects a shape of (None,32,32,3) and hence we turn [img] into an np array
```
- This code can be found in `tester.py`

## Final note
- A lot of the architectural decisions were found out by trial and error, and at the end the model only has a maximum accuracy of around 75-76%