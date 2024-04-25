# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.
## Neural Network Model
![image](https://github.com/janani225/mnist-classification/assets/113497333/b9f4a064-d878-4c88-aef9-813fb159cfcb)




## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing.

### STEP 2:
Build a CNN model.
### STEP 3:
Compile and fit the model and then predict.

## PROGRAM

### Name:BS.Amrutha Varshni
### Register Number:212222040007


```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import keras as kf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[59999]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add (layers. Input (shape=(28,28,1)))
model.add (layers. Conv2D (filters=32, kernel_size=(5,5), activation='relu'))
model.add (layers. MaxPool2D (pool_size=(2,2)))
model.add (layers. Flatten())
model.add (layers. Dense (32, activation='relu'))
model.add (layers. Dense (16, activation='relu'))
model.add (layers. Dense (8, activation='relu'))
model.add (layers. Dense (10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train_scaled,y_train_onehot,epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled, y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(classification_report(y_test,x_test_predictions))
img = image.load_img('deep.png')
type(img)
img = image.load_img('deep.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/Amruthavarshnibs/mnist-classification/assets/119103704/9b6e6a5b-cc2d-4c24-b164-3252184d7a12)

![image](https://github.com/Amruthavarshnibs/mnist-classification/assets/119103704/0240b1c2-914a-4fa6-98bc-004094d7cf06)

### Classification Report

![image](https://github.com/Amruthavarshnibs/mnist-classification/assets/119103704/3d490982-9b1a-4113-9acd-008643ecca8e)




### Confusion Matrix
![image](https://github.com/Amruthavarshnibs/mnist-classification/assets/119103704/2e7dce92-0af9-4d7a-8ccf-0f0f72622d30)





### New Sample Data Prediction

![image](https://github.com/Amruthavarshnibs/mnist-classification/assets/119103704/f912f686-f012-4165-8cb2-3fc31c030c19)



## RESULT
A Convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
