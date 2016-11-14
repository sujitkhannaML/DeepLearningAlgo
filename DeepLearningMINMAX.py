import numpy as np
import scipy
import matplotlib as plt
import pandas as pd
import datetime as dt
import math
import sklearn
from sklearn import cross_validation, metrics
from sklearn.linear_model import LogisticRegression
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



######Set random seed for reproducable Results #######

np.random.seed(24)

#####Read inputs Features and Output Classes######
dataframe=pd.read_csv()
dataframe1=np.asarray(dataframe).astype(object)
datetime=dataframe1[:,0:3]
dataset=dataframe1[:,3:13]
dataset=dataset.astype('float32')

#####Initiate Inputs and outputs#########################
price,bsratio,bssize,depth,bavol,ofi,oir,posdev,negdev,futprice=dataset[:,0],dataset[:,1],dataset[:,2],dataset[:,3],dataset[:,4],dataset[:,5],dataset[:,6],dataset[:,7],dataset[:,8],dataset[:,9]

####transform data into a stationary series########
scaler=MinMaxScaler(feature_range=(0,1))
#scaler.fit_transform(price)
price1,bsratio1,bssize1,depth1=scaler.fit_transform(price),scaler.fit_transform(bsratio),scaler.fit_transform(bssize)scaler.fit_transform(depth)
bavol1,ofi1,oir1=scaler.fit_transform(bavol),scaler.fit_transform(ofi),scaler.fit_transform(oir)
posdev1,negdev1,futprice1=scaler.fit_transform(posdev),scaler.fit_transform(negdev),scaler.fit_transform(futprice)

train_size=int(len(dataset)*0.8)
test_size=len(dataset)-train_size

#####define final ip/op sets here
final_ip=np.vstack((price1,bsratio1,bssize1,depth1,bavol1,ofi1,oir1,posdev1,negdev1)).T
final_op=futprice1
train_ip,train_op=final_ip[0:train_size,:],final_op[0:train_size]
test_ip,test_op=final_ip[train_size:len(final_ip),:],final_op[train_size:len(final_ip)]

######Reshape IPs in the form of [sample, time steps, features]######
train_ipr=np.reshape(train_ip,(train_ip.shape[0], 1, train_ip.shape[1]))
test_ipr=np.reshape(test_ip,(test_ip.shape[0], 1, test_ip.shape[1]))

######define and fir Deep Learning Model###################
dimensions=train_ip.shape[1]
model=Sequential()
model.add(LSTM(10, input_dim=dimensions, activation='tanh',return_sequences=True))
model.add(LSTM(10, input_dim=dimensions, activation='tanh',return_sequences=False))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_ipr, train_op, nb_epoch=100, batch_size=1)
#add dropout for model refinement######

######in/out sample prediction#######################
trainPredict=model.predict(train_ipr)
testPredict=model.predict(test_ipr)

traininv=scaler.inverse_transform(trainPredict)
trainop=scaler.inverse_transform(train_op)
testinv=scaler.inverse_transform(testPredict)
testop=scaler.inverse_transform(test_op)

######calculate root mean square errors########
trainScore=math.sqrt(mean_squared_error(trainop[0],traininv[:,0]))
print('Train Score:%.2f RMSE' % (trainScore))

testScore=math.sqrt(mean_squared_error(testop[0],testinv[:,0]))
print('Train Score:%.2f RMSE' % (testScore))

####Plot performances##########################
testPredictPlot=np.empty_like(fut_price1)
testPredictPlot[:]=np.nan
testPredictPlot[len(trainPredict):len(futprice1)]=testinv

plt.plot(scaler.inverse_transform(futprice1))
plt.matplotlib_fname(traininv)
plt.plot(testPredictPlot)
plt.show()

################Export to excel here##########
osseries=(testinv).T
allop=np.vstack(test_op,osseries)
allop1=pd.DataFrame(allop)
