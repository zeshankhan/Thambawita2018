# -*- coding: utf-8 -*-
"""
zeshan khan
"""


features_dense='features_densenet_train.csv'
features_denset='features_densenet_test.csv'
features_res='features_resnet_train.csv'
features_rest='features_resnet_test.csv'

import pandas as pd
import numpy as np

df1=pd.read_csv(features_dense,header=None)
df2=pd.read_csv(features_res,header=None)
df3=pd.read_csv(features_denset,header=None)
df4=pd.read_csv(features_rest,header=None)

actual1=df1.iloc[:,-3]
actual2=df2.iloc[:,-3]
actual3=df3.iloc[:,-3]
actual4=df4.iloc[:,-3]
pred_prob1=df1.iloc[:,:-3]
pred_prob2=df2.iloc[:,:-3]
pred_prob3=df3.iloc[:,:-3]
pred_prob4=df4.iloc[:,:-3]
pred1=pred_prob1.idxmax(axis = 1)+1
pred2=pred_prob2.idxmax(axis = 1)+1
pred3=pred_prob3.idxmax(axis = 1)+1
pred4=pred_prob4.idxmax(axis = 1)+1

X = pd.concat([pred_prob1, pred_prob1], axis=1, join='inner')
Xt = pd.concat([pred_prob3, pred_prob4], axis=1, join='inner')

actual16=np.zeros((X.shape[0],16),int)
actual16t=np.zeros((Xt.shape[0],16),int)

for i in range(X.shape[0]):
  actual16[i,actual1[i]-1]=1

for i in range(Xt.shape[0]):
  actual16t[i,actual3[i]-1]=1

# Network Building

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split

model = keras.Sequential()
model.add(layers.Dense(32, activation="relu", name="L1"))
model.add(layers.Dense(16, activation="softmax", name="L2"))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

X_train, X_test, Y_train, Y_test = train_test_split(X, actual16, test_size=0.33, random_state=5)
#400,92,90
model.fit(
    X_train,
    Y_train,
    batch_size=25,
    epochs=500,
    validation_data=(X_test, Y_test),
)

pred_prob=pd.DataFrame(model.predict(Xt))
pred=pred_prob.idxmax(axis = 1)+1

"""
Some evaluations

actualf=pd.DataFrame(actual16t).idxmax(axis = 1)+1
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
acc=accuracy_score(actualf,pred)
f1=f1_score(actualf,pred,average='weighted')
print(acc,f1)
"""
