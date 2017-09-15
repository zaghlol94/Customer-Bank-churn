#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:19:37 2017

@author: zaghlol
"""

import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#handle cat data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
x[:, 1] = labelencoder_X_1.fit_transform(x[:, 1])

labelencoder_X_2 = LabelEncoder()
x[:, 2] = labelencoder_X_1.fit_transform(x[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
x= onehotencoder.fit_transform(x).toarray()

x=x[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size = 0.20, random_state = 0)


#feat scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)

#ANN
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier=Sequential()

classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

classifier.add(Dropout(p=0.1))

classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(Xtrain,Ytrain,batch_size=10,nb_epoch=100)

Ypred=classifier.predict(Xtest)

Ypred=(Ypred>0.5)

#confusion matrix
from sklearn.metrics import confusion_matrix
cn=confusion_matrix(Ytest,Ypred)

NewPrediction=classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,600,2,1,1,5000]])))
NewPrediction=(NewPrediction>0.5)

#evaluate
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier=Sequential()
    
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    return classifier

classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,nb_epoch=100)

acc=cross_val_score(estimator=classifier,X=Xtrain,y=Ytrain,cv=10,n_jobs=-1)


mean=acc.mean()
variance=acc.std()


#using GridSearch

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(opt):
    classifier=Sequential()
    
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    
    classifier.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])

    return classifier

classifier=KerasClassifier(build_fn=build_classifier)

parameters ={'batch_size':[25,32],
            'nb_epoch':[100,500],
            'opt':['adam','rmsprop']}

GS=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)

GS=GS.fit(Xtrain,Ytrain)

best_params=GS.best_params_
best_acc=GS.best_score_