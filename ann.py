# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13 ].values
y = dataset.iloc[:, 13].values

# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# X =X[:,1:]
onehotencoder = ColumnTransformer([("onehotencoder",OneHotEncoder(),[1])],remainder='passthrough')
X = onehotencoder.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#part-2 - Now let's make the ANN!

#Importing the Keras Libraries and Packages

import keras 
from keras.models import Sequential
from keras.layers import Dense

#Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6,init ='uniform',activation= 'relu',input_dim =11))
#Adding the second hidden layer
classifier.add(Dense(output_dim=6,init = 'uniform', activation ='relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1,init ='uniform',activation= 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer ='adam' ,loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the ANN to the training set
classifier.fit(X_train,y_train,batch_size = 10 ,nb_epoch =100)

#Part 3- Making the predictions and evaluating the model


#Predicting the Test set results

# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
