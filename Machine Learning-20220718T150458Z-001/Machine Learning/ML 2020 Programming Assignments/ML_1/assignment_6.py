# -*- coding: utf-8 -*-
"""Assignment_6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zJbqJaSFhZ0A9K7GD2yxuVc7FHglWHPP

# **Install necessary libraries**
"""

# Commented out IPython magic to ensure Python compatibility.
!pip install tensorflow keras numpy mnist matplotlib
import numpy as np
import pandas as pd
import mnist
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
# %matplotlib inline

"""# **Split the dataset into train, test and validation**"""

X_train=mnist.train_images()
Y_train=mnist.train_labels()
X_val  = X_train[50000:60000]
X_train = X_train[0:50000]
Y_val  = Y_train[50000:60000]
Y_train = Y_train[0:50000]
X_test=mnist.test_images()
Y_test=mnist.test_labels()

"""# **Normalize pixels[0,255] into [-0.5,0.5] and Reshaping it**"""

X_train= (X_train/255)-0.5
X_test= (X_test/255)-0.5
X_val= (X_val/255)-0.5
X_train= X_train.reshape(-1,784)
X_test= X_test.reshape(-1,784)
X_val= X_val.reshape(-1,784)

"""## **Building model and using relu and sigmoid function**"""

# Adding 1 hidden layers with 64 neurons with activation function=relu and sigmoid
# 1 layer with 10 neurons with softmax function
model = Sequential()
model.add(Dense(64, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))

"""# **Compiling and Fitting(backround black and digit white) with batch size 100 and epochs chosen 30,30,20,33 for four different cases described in report**"""

# Adam optimizer improves the model 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
h=model.fit(X_train, to_categorical(Y_train),validation_split=0.16,batch_size= 100, epochs =30,verbose = 1)

"""**Evaluate the model**"""

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Case 1:model loss with batch size 100')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

s=model.evaluate(X_test, to_categorical(Y_test))
print("accuracy: ", s[1])

"""#Case 2: Model with 1 hidden layer with 32 neurons and relu as activation function#"""

# Adding 1 hidden layers with 32 neurons with activation function=relu and sigmoid
# 1 layer with 10 neurons with softmax function
model = Sequential()
model.add(Dense(32, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))

#Case 2:
# Adam optimizer improves the model 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
h=model.fit(X_train, to_categorical(Y_train),validation_split=0.16,batch_size= 100, epochs =30,verbose = 1)

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Case 2:model loss with batch size 100')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

s=model.evaluate(X_test, to_categorical(Y_test))
print("accuracy: ", s[1])

"""#Case 3: Model with 2 hidden layer with 64 neurons each with relu and sigmoid as activation functions#"""

# Adding 2 hidden layers with 64 neurons with activation function=relu and sigmoid
# 1 layer with 10 neurons with softmax function
model = Sequential()
model.add(Dense(64, input_dim=784, activation='relu'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

#Case 3:
# Adam optimizer improves the model 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
h=model.fit(X_train, to_categorical(Y_train),validation_split=0.16,batch_size= 100, epochs =20,verbose = 1)

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Case 3:model loss with batch size 100')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

s=model.evaluate(X_test, to_categorical(Y_test))
print("accuracy: ", s[1])

"""#Case 4: Model with 2 hidden layer with 32 neurons each with relu and sigmoid as activation functions.#"""

# Adding 2 hidden layers with 32 neurons with activation function=relu and sigmoid
# 1 layer with 10 neurons with softmax function
model = Sequential()
model.add(Dense(32, input_dim=784, activation='relu'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

#Case 4:
# Adam optimizer improves the model 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
h=model.fit(X_train, to_categorical(Y_train),validation_split=0.16,batch_size= 100, epochs =33,verbose = 1)

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Case 4:model loss with batch size 100')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

s=model.evaluate(X_test, to_categorical(Y_test))
print("accuracy: ", s[1])

