import cv2
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('horse.jpg',cv2.IMREAD_COLOR)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(32,32))

labels = ["Plane","Car","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck"]
model = models.load_model('75_model.h5')
prediction = model.predict(np.array([img]))

plt.imshow(img)
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.title(labels[prediction])
plt.show()