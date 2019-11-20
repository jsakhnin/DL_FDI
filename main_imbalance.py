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

def imbalance_data(sysName, imbalance):
    X=[]
    y = []
    
    for i in range(1,11,1):
        print (i)
        X.append(pd.read_csv("Data/{}/{}Data_10k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None).iloc[:int(5000+round(imbalance*5000))] )
        y.append(pd.read_csv("Data/{}/{}Labels_10k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None).iloc[:int(5000+round(imbalance*5000))])
        
        X[i-1],y[i-1] = dataProc(X[i-1],y[i-1])
    
        
    X_train = np.concatenate((X[0],X[1]), axis=0)
    
    y_train = np.concatenate((y[0],y[1]), axis=0)
    for i in range(2,10,1):
        X_train = np.concatenate((X_train,X[i]), axis=0) 
        y_train = np.concatenate((y_train,y[i]), axis=0)
    
    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
    
    return X_train, X_val, y_train, y_val

def load_testData(sysName):
    ##Loading Test Data
    Xt = []
    yt = []
    
    for i in range(1,11,1):
        print (i)
        Xt.append(pd.read_csv("Data/{}/{}Data_2k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None))
        yt.append(pd.read_csv("Data/{}/{}Labels_2k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None))
        
        Xt[i-1],yt[i-1] = dataProc(Xt[i-1],yt[i-1])
    
    return Xt, yt
##################################################################################
sysName = "IEEE57"
numEpochs = 100
testType = "MainModels_Imbalanced"
##########################################################################################################
#################################             Deep Learning                          #####################
imbalanceRange = np.arange(0.1,1.0,0.1)
Results = []

Xt, yt = load_testData(sysName)
sparsity = np.arange(0.1,1.1,0.1)

for imbalance in imbalanceRange:
    
    checkpoint_path = "Saved Models/"+sysName+"_models/{}".format(imbalance)
    X_train, X_val, y_train, y_val = imbalance_data(sysName, imbalance)
    numFeatures = len(X_train[0])
    print("Features = ", numFeatures)    
    #Model with no regulation
    model1 = m.DLmodel1(numFeatures)
    LOGNAME = "IMBALANCED{}---{}-{}-model1-{}Epochs-{}".format(imbalance,sysName,testType , numEpochs, int(time.time()) )
    tensorboard = TensorBoard(log_dir='logs\{}'.format(LOGNAME))
    history1 = model1.fit(X_train,y_train,epochs=numEpochs ,batch_size=32,validation_data=(X_val,y_val), callbacks = [tensorboard])
    model1.save(checkpoint_path+'model1_{}.h5'.format(numEpochs))
    result1 = evaluateModel(model1, Xt, yt)
    
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
      monitor='val_acc', min_delta=0.0001,
      patience=5)
    
    model7 = m.DLmodel7(numFeatures)
    LOGNAME = "{}-{}-model7-{}Epochs-{}".format(sysName,testType , numEpochs, int(time.time()) )
    tensorboard = TensorBoard(log_dir='logs\{}'.format(LOGNAME))
    history7 = model7.fit(X_train,y_train,epochs=numEpochs ,batch_size=32,validation_data=(X_val,y_val), callbacks = [tensorboard, earlystop_callback])
    model7.save(checkpoint_path+'model7_EarlyStop.h5')
    result7 = evaluateModel(model7,Xt,yt)

    Results.append(  pd.DataFrame({'Sparsity': sparsity, 'Model 1': result1, 'Model 7': result7})  )


################   DATA OUTPUT (Saving in Excel)    ###############
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('RESULTS_'+sysName+'_'+testType+'_'+str(numEpochs)+'Epochs.xlsx', engine='xlsxwriter') #CHANGE THE NAME OF THE OUTPUT EXCEL FILE HERE

for i in range(9):
    Results[i].to_excel(writer, sheet_name = "Imb {}".format(imbalanceRange[i]) )


# Close the Pandas Excel writer and output the Excel file.
writer.save()

print("PROGRAM IS COMPLETE !!!!! ")
