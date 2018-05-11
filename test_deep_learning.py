# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:08:05 2018

@author: rachitjo960
"""
#import libarires
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout
import pandas
import numpy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV

#dataset reading
dataset = pandas.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
df_x = pandas.DataFrame(x)

#encoding
x_labelencoder = LabelEncoder()
x[:, 1] = x_labelencoder.fit_transform(x[:, 1])
x[:, 2] = x_labelencoder.fit_transform(x[:, 2])
onehot = OneHotEncoder(categorical_features = [1])
x = onehot.fit_transform(x).toarray()
#avoid dummy vairable trap
x = x[:, 1:]

#split in train and test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#feature_scaling
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.fit_transform(x_test)

#design ANN
classifier = Sequential()

#build hidden layers
classifier.add(Dense(units =6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(units =6, kernel_initializer = 'uniform', activation = 'relu'))

# build output layer
classifier.add(Dense(units =1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#compile ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#applyting ANN
classifier.fit(x_train,y_train,batch_size=10,epochs=50)

# prediction
y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)

#confusion matrix
cm = confusion_matrix(y_test, y_pred)

# predicting for single observation
'''
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
'''
single_pred = classifier.predict(scale.transform(numpy.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))

#evaluation
#K-fold cross validation
'''
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units =6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units =6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units =1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size =10, epochs = 50)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv =10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()
'''
'''
# Tuning the ANN
def build_classifier(opt):
    classifier = Sequential()
    classifier.add(Dense(units =6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units =6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units =1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier    
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size' : [25,32],
              'epochs' : [50,100],
              'optimizer' : ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, cv =10, scoring = 'accuracy')
grid_search = grid_search.fit(x_train, y_train)
best_param= grid_search.best_params_
best_acc= grid_search.best_score_
'''






















