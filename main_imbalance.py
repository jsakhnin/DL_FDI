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
numEpochs = 100
imbalance = 0.5 #percentage of attack data to keep
testType = "MainModels_Imbalanced_{}".format(imbalance)

X=[]
Xt = []
y = []
yt = []

for i in range(1,11,1):
    print (i)
    X.append(pd.read_csv("Data/{}/{}Data_10k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None).iloc[:(5000+round(imbalance*5000))] )
    Xt.append(pd.read_csv("Data/{}/{}Data_2k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None).iloc[:(5000+round(imbalance*5000))] )
    y.append(pd.read_csv("Data/{}/{}Labels_10k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None).iloc[:(5000+round(imbalance*5000))])
    yt.append(pd.read_csv("Data/{}/{}Labels_2k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None).iloc[:(5000+round(imbalance*5000))])
    
    X[i-1],y[i-1] = dataProc(X[i-1],y[i-1])
    Xt[i-1],yt[i-1] = dataProc(Xt[i-1],yt[i-1])

    
X_train = np.concatenate((X[0],X[1]), axis=0)
X_test = np.concatenate((Xt[0],Xt[1]), axis=0)

y_train = np.concatenate((y[0],y[1]), axis=0)
y_test = np.concatenate((yt[0],yt[1]), axis=0)
numFeatures = len(X_train[0])
for i in range(2,10,1):
    X_train = np.concatenate((X_train,X[i]), axis=0) 
    X_test = np.concatenate((X_test,Xt[i]), axis=0) 
    y_train = np.concatenate((y_train,y[i]), axis=0)
    y_test = np.concatenate((y_test,yt[i]), axis=0)

X_train, y_train = shuffle(X_train, y_train, random_state=0)
X_test, y_test = shuffle(X_test, y_test, random_state=0)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

print("Training Data: ", X_train.shape, " ", y_train.shape)
print("Validation Data: ", X_val.shape, " ", y_val.shape)
print("Test Data: ", X_test.shape, " ", y_test.shape)
##########################################################################################################
#################################             Deep Learning                          #####################

checkpoint_path = "Saved Models/"+sysName+"_models/".format(imbalance)

#Model with no regulation
model1 = m.DLmodel1(numFeatures)
LOGNAME = "IMBALANCED{}---{}-{}-model1-{}Epochs-{}".format(imbalance,sysName,testType , numEpochs, int(time.time()) )
tensorboard = TensorBoard(log_dir='logs\{}'.format(LOGNAME))
history1 = model1.fit(X_train,y_train,epochs=numEpochs ,batch_size=32,validation_data=(X_val,y_val), callbacks = [tensorboard])
model1.save(checkpoint_path+'model1_{}.h5'.format(numEpochs))
result1 = evaluateModel(model1, Xt, yt)

model7 = m.DLmodel7(numFeatures)
LOGNAME = "IMBALANCED{}---{}-{}-model7-{}Epochs-{}".format(imbalance,sysName,testType , numEpochs, int(time.time()) )
tensorboard = TensorBoard(log_dir='logs\{}'.format(LOGNAME))
history7 = model7.fit(X_train,y_train,epochs=numEpochs ,batch_size=32,validation_data=(X_val,y_val), callbacks = [tensorboard])
model7.save(checkpoint_path+'model7_{}.h5'.format(numEpochs))
result7 = evaluateModel(model7,Xt,yt)

sparsity = np.arange(0.1,1.1,0.1)


################   DATA OUTPUT (Saving in Excel)    ###############
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('RESULTS_'+sysName+'_'+testType+'_'+str(numEpochs)+'Epochs.xlsx', engine='xlsxwriter') #CHANGE THE NAME OF THE OUTPUT EXCEL FILE HERE

#Results = pd.DataFrame({'Sparsity': sparsity, 'Model 1': result1, 'Model 2': result2,'Model 3': result3, 'Model 4': result4,
                        #'Model 5': result5,  'Model 6': result6, 'Model 7': result7})

Results = pd.DataFrame({'Sparsity': sparsity, 'Model 1': result1, 'Model 7': result7})

# Convert the dataframe to an XlsxWriter Excel object.
Results.to_excel(writer, sheet_name=sysName)

# Close the Pandas Excel writer and output the Excel file.
writer.save()

print("PROGRAM IS COMPLETE !!!!! ")
