import sys
#sys.path.append('C:\\Jacob\\pip')
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn import metrics, svm
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
import ModelsFinal as m
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import TensorBoard

################################################################################
def dataProc(X1,y1):
    X1 = pd.DataFrame(X1)
    X1 = X1.fillna(X1.mean())
    X1=X1.to_numpy()
    y1 = y1.to_numpy()
    
    
    min_max_scaler = preprocessing.MinMaxScaler()
    X1 = min_max_scaler.fit_transform(X1)
    
    X1, y1 = shuffle(X1, y1, random_state=0)
    return X1,y1

def evaluateModel(model,Xt,yt):
    result = []
    
    for i in range(10):
        resultTemp = model.evaluate(Xt[i], yt[i])
        print(resultTemp)
        result.append(resultTemp[1])
    return result

def plot_history(histories, key='acc'):
    plt.figure(figsize=(16,10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    return plt

##################################################################################
sysName = "IEEE30"
testType = "MainModelsTest"
numEpochs = 100

X=[]
Xv = []
Xt = []
y = []
yv = []
yt = []

for i in range(1,11,1):
    print (i)
    X.append(pd.read_csv("Data/{}/{}Data_10k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None))
    Xv.append(pd.read_csv("Data/{}/{}Data_3k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None))
    Xt.append(pd.read_csv("Data/{}/{}Data_2k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None))
    y.append(pd.read_csv("Data/{}/{}Labels_10k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None))
    yv.append(pd.read_csv("Data/{}/{}Labels_3k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None))
    yt.append(pd.read_csv("Data/{}/{}Labels_2k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None))
    
    X[i-1],y[i-1] = dataProc(X[i-1],y[i-1])
    Xt[i-1],yt[i-1] = dataProc(Xt[i-1],yt[i-1])

    


X_train = np.concatenate((X[0],X[1]), axis=0)
X_val = np.concatenate((Xv[0],Xv[1]), axis=0)
X_test = np.concatenate((Xt[0],Xt[1]), axis=0)

y_train = np.concatenate((y[0],y[1]), axis=0)
y_val = np.concatenate((yv[0],yv[1]), axis=0)
y_test = np.concatenate((yt[0],yt[1]), axis=0)
numFeatures = len(X_train[0])
for i in range(2,10,1):
    X_train = np.concatenate((X_train,X[i]), axis=0) 
    X_val = np.concatenate((X_val,Xv[i]), axis=0) 
    X_test = np.concatenate((X_test,Xt[i]), axis=0) 
    y_train = np.concatenate((y_train,y[i]), axis=0)
    y_val = np.concatenate((y_val,yv[i]), axis=0)
    y_test = np.concatenate((y_test,yt[i]), axis=0)

X_train, y_train = shuffle(X_train, y_train, random_state=0)
X_val, y_val = shuffle(X_val, y_val, random_state=0)
X_test, y_test = shuffle(X_test, y_test, random_state=0)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

print("Training Data: ", X_train.shape, " ", y_train.shape)
print("Validation Data: ", X_val.shape, " ", y_val.shape)
print("Test Data: ", X_test.shape, " ", y_test.shape)
##########################################################################################################
#################################             Deep Learning                          #####################   
checkpoint_path = "Saved Models/"+sysName+"_models/"

#Model with no regulation
model1 = m.DLmodel1(numFeatures)
LOGNAME = "{}-{}-model1-{}Epochs-{}".format(sysName,testType , numEpochs, int(time.time()) )
tensorboard = TensorBoard(log_dir='logs\{}'.format(LOGNAME))
history1 = model1.fit(X_train,y_train,epochs=numEpochs ,batch_size=32,validation_data=(X_val,y_val), callbacks = [tensorboard])
model1.save(checkpoint_path+'model1.h5')
result1 = evaluateModel(model1, Xt, yt)


model2 = m.DLmodel2(numFeatures)
LOGNAME = "{}-{}-model2-{}Epochs-{}".format(sysName,testType , numEpochs, int(time.time()) )
tensorboard = TensorBoard(log_dir='logs\{}'.format(LOGNAME))
history2 = model2.fit(X_train,y_train,epochs=numEpochs ,batch_size=32,validation_data=(X_val,y_val), callbacks = [tensorboard])
model2.save(checkpoint_path+'model2_{}.h5'.format(numEpochs))
result2 = evaluateModel(model2,Xt,yt)


model3 = m.DLmodel3(numFeatures)
LOGNAME = "{}-{}-model3-{}Epochs-{}".format(sysName,testType , numEpochs, int(time.time()) )
tensorboard = TensorBoard(log_dir='logs\{}'.format(LOGNAME))
history3 = model3.fit(X_train,y_train,epochs=numEpochs ,batch_size=32,validation_data=(X_val,y_val), callbacks = [tensorboard])
model3.save(checkpoint_path+'model3_{}.h5'.format(numEpochs))
result3 = evaluateModel(model3,Xt,yt)


model4 = m.DLmodel4(numFeatures)
LOGNAME = "{}-{}-model4-{}Epochs-{}".format(sysName,testType , numEpochs, int(time.time()) )
tensorboard = TensorBoard(log_dir='logs\{}'.format(LOGNAME))
history4 = model4.fit(X_train,y_train,epochs=numEpochs ,batch_size=32,validation_data=(X_val,y_val), callbacks = [tensorboard])
model4.save(checkpoint_path+'model4_{}.h5'.format(numEpochs))
result4 = evaluateModel(model4,Xt,yt)


model5 = m.DLmodel5(numFeatures)
LOGNAME = "{}-{}-model5-{}Epochs-{}".format(sysName,testType , numEpochs, int(time.time()) )
tensorboard = TensorBoard(log_dir='logs\{}'.format(LOGNAME))
history5 = model5.fit(X_train,y_train,epochs=numEpochs ,batch_size=32,validation_data=(X_val,y_val), callbacks = [tensorboard])
model5.save(checkpoint_path+'model5_{}.h5'.format(numEpochs))
result5 = evaluateModel(model5,Xt,yt)


model6 = m.DLmodel6(numFeatures)
LOGNAME = "{}-{}-model6-{}Epochs-{}".format(sysName,testType , numEpochs, int(time.time()) )
tensorboard = TensorBoard(log_dir='logs\{}'.format(LOGNAME))
history6 = model6.fit(X_train,y_train,epochs=numEpochs ,batch_size=32,validation_data=(X_val,y_val), callbacks = [tensorboard])
model6.save(checkpoint_path+'model6_{}.h5'.format(numEpochs))
result6 = evaluateModel(model6,Xt,yt)

sparsity = np.arange(0.1,1.1,0.1)

################   DATA OUTPUT (Saving in Excel)    ###############
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('RESULTS_'+sysName+'_'+testType+'_'+str(numEpochs)+'Epochs.xlsx', engine='xlsxwriter') #CHANGE THE NAME OF THE OUTPUT EXCEL FILE HERE

Results = pd.DataFrame({'Sparsity': sparsity, 'Model 1': result1, 'Model 2': result2,'Model 3': result3, 'Model 4': result4,
                        'Model 5': result5,  'Model 6': result6})

# Convert the dataframe to an XlsxWriter Excel object.
Results.to_excel(writer, sheet_name=sysName)

# Close the Pandas Excel writer and output the Excel file.
writer.save()

print("PROGRAM IS COMPLETE !!!!! ")
