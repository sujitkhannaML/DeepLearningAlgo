import numpy as np
import scipy
import matplotlib as plt
import pandas as pd
import datetime as dt
import math
import sklearn
from sklearn import cross_validation, metrics
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale,MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import theano
import theano.tensor as T
import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM,Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils


#####Set random seed for reproducable Results #######

np.random.seed(24)


#####Read inputs Features and Output Classes######
dataframe=pd.read_csv('C:\\Users\\sujit\\Desktop\\python\\DeepLearning\\NIFTYFEATURES.CSV')
dataframe1=np.asarray(dataframe).astype(object)
datetime=dataframe1[:,0:3]
dataset=dataframe1[:,3:13]
dataset=dataset.astype('float32')

#####Initiate Inputs and outputs#########################
######inputs and outputs are read as deciles of rangine from  1-10###################
price,bsratio,bssize,depth,bavol,ofi,oir,posdev,negdev,futprice=dataset[:,0],dataset[:,1],dataset[:,2],dataset[:,3],dataset[:,4],dataset[:,5],dataset[:,6],dataset[:,7],dataset[:,8],dataset[:,9]
train_size=int(len(dataset)*0.8)
test_size=len(dataset)-train_size

#####define final ip/op sets here
final_ip=np.vstack((price,bsratio,bssize,depth,bavol,ofi,oir,posdev,negdev)).T
final_op=futprice

##########Encode Class values as integers############################
encoder=LabelEncoder()
encoder.fit(final_op)
encoded_Y=encoder.transform(final_op)
dummy_Y=np_utils.to_categorical(encoded_Y)

train_ip,train_op=final_ip[0:train_size,:],dummy_Y[0:train_size]
test_ip,test_op=final_ip[train_size:len(final_ip),:],dummy_Y[train_size:len(final_ip)]

#########Define RNN Model here#####################
dimensions=train_ip.shape[1]
model=Sequential()
model.add(LSTM(10, input_dim=dimensions, activation='tanh',return_sequences=True))
model.add(LSTM(10, input_dim=dimensions, activation='tanh',return_sequences=False))
model.add(Dense(10,init='normal',activation='tanh'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_ip, train_op, nb_epoch=100, batch_size=1)
#add dropout for model refinement######

######in/out sample prediction#######################
trainPredict=model.predict(train_ipr)
testPredict=model.predict(test_ipr)

####Plot performances##########################
testPredictPlot=np.empty_like(fut_price1)
testPredictPlot[:]=np.nan
testPredictPlot[len(trainPredict):len(futprice1)]=testinv

plt.plot(scaler.inverse_transform(futprice1))
plt.matplotlib_fname(trainPredict)
plt.plot(testPredictPlot)
plt.show()

################Export to excel here##########
osseries=(testinv).T
allop=np.vstack(test_op,osseries)
allop1=pd.DataFrame(allop)
allop1.to_csv('C:/Users/sujit/Desktop/python/DeepLearning/output.csv')








