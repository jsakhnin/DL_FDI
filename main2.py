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
    X1=X1.to_numpy()
    y1 = y1.to_numpy()
    
    
    min_max_scaler = preprocessing.MinMaxScaler()
    X1 = min_max_scaler.fit_transform(X1)
    
    X1, y1 = shuffle(X1, y1, random_state=0)
    return X1,y1

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
sysName = "IEEE14"
testType = "MainModels"
numEpochs = 100
checkpoint_path = "Saved Models/"+sysName+"_models/"

X1 = pd.read_csv("Data/"+sysName+"/"+sysName+"Data_40k_lowSparsity"+".csv")
y1 = pd.read_csv("Data/"+sysName+"/"+sysName+"Labels_40k_lowSparsity"+".csv")
X2 = pd.read_csv("Data/"+sysName+"/"+sysName+"Data_40k_midSparsity"+".csv")
y2 = pd.read_csv("Data/"+sysName+"/"+sysName+"Labels_40k_midSparsity"+".csv")
X3 = pd.read_csv("Data/"+sysName+"/"+sysName+"Data_40k_highSparsity"+".csv")
y3 = pd.read_csv("Data/"+sysName+"/"+sysName+"Labels_40k_highSparsity"+".csv")

Xv1 = pd.read_csv("Data/"+sysName+"/"+sysName+"Data_10k_lowSparsity"+".csv")
yv1 = pd.read_csv("Data/"+sysName+"/"+sysName+"Labels_10k_lowSparsity"+".csv")
Xv2 = pd.read_csv("Data/"+sysName+"/"+sysName+"Data_10k_midSparsity"+".csv")
yv2 = pd.read_csv("Data/"+sysName+"/"+sysName+"Labels_10k_midSparsity"+".csv")
Xv3 = pd.read_csv("Data/"+sysName+"/"+sysName+"Data_10k_highSparsity"+".csv")
yv3 = pd.read_csv("Data/"+sysName+"/"+sysName+"Labels_10k_highSparsity"+".csv")

Xt1 = pd.read_csv("Data/"+sysName+"/"+sysName+"Data_4k_lowSparsity"+".csv")
yt1 = pd.read_csv("Data/"+sysName+"/"+sysName+"Labels_4k_lowSparsity"+".csv")
Xt2 = pd.read_csv("Data/"+sysName+"/"+sysName+"Data_4k_midSparsity"+".csv")
yt2 = pd.read_csv("Data/"+sysName+"/"+sysName+"Labels_4k_midSparsity"+".csv")
Xt3 = pd.read_csv("Data/"+sysName+"/"+sysName+"Data_4k_highSparsity"+".csv")
yt3 = pd.read_csv("Data/"+sysName+"/"+sysName+"Labels_4k_highSparsity"+".csv")

X1,y1 = dataProc(X1,y1)
Xv1,yv1 = dataProc(Xv1,yv1)
Xt1,yt1 = dataProc(Xt1,yt1)
X2,y2 = dataProc(X2,y2)
Xv2,yv2 = dataProc(Xv2,yv2)
Xt2,yt2 = dataProc(Xt2,yt2)

X3,y3 = dataProc(X3,y3)
Xv3,yv3 = dataProc(Xv3,yv3)
Xt3,yt3 = dataProc(Xt3,yt3)

numFeatures = len(X1[0])

X = np.concatenate((X1, X2, X3), axis=0)
y = np.concatenate((y1, y2, y3), axis=0)
Xv = np.concatenate((Xv1, Xv2, Xv3), axis=0)
yv = np.concatenate((yv1, yv2, yv3), axis=0)
Xt = np.concatenate((Xt1, Xt2, Xt3), axis=0)
yt = np.concatenate((yt1, yt2, yt3), axis=0)

X, y = shuffle(X, y, random_state=0)
Xv, yv = shuffle(Xv, yv, random_state=0)
Xt, yt = shuffle(Xt, yt, random_state=0)

##########################################################################################################
#################################             Deep Learning                          #####################   
#PROPOSED MODEL
model1 = m.DLmodel1(numFeatures)
LOGNAME = "{}-model1-{}".format(testType , int(time.time()) )
tensorboard = TensorBoard(log_dir='logs\{}'.format(LOGNAME))
history1 = model1.fit(X,y,epochs=numEpochs ,batch_size=32,validation_data=(Xv,yv), callbacks = [tensorboard])
results11 = model1.evaluate(Xt1, yt1)
results12 = model1.evaluate(Xt2, yt2)
results13 = model1.evaluate(Xt3, yt3)
model1.save(checkpoint_path+'model1.h5')

#OTHER MODEL
model2 = m.DLmodel2(numFeatures)
LOGNAME = "{}-model2-{}".format(testType , int(time.time()) )
tensorboard = TensorBoard(log_dir='logs\{}'.format(LOGNAME))
history2 = model2.fit(X,y,epochs=numEpochs ,batch_size=32,validation_data=(Xv,yv), callbacks = [tensorboard])
results21 = model2.evaluate(Xt1, yt1)
results22 = model2.evaluate(Xt2, yt2)
results23 = model2.evaluate(Xt3, yt3)
model2.save(checkpoint_path+'model2.h5')

#OTHER MODEL
model3 = m.DLmodel3(numFeatures)
LOGNAME = "{}-model3-{}".format(testType , int(time.time()) )
tensorboard = TensorBoard(log_dir='logs\{}'.format(LOGNAME))
history3 = model3.fit(X,y,epochs=numEpochs ,batch_size=32,validation_data=(Xv,yv), callbacks = [tensorboard])
results31 = model3.evaluate(Xt1, yt1)
results32 = model3.evaluate(Xt2, yt2)
results33 = model3.evaluate(Xt3, yt3)
model3.save(checkpoint_path+'model3.h5')
################################       Final Data Processing            #################################
results_low = [results11[1], results21[1], results31[1]]
results_mid = [results12[1], results22[1], results32[1]]
results_high = [results13[1], results23[1], results33[1]]

names = [  'Model 1', 
               'Model 2',
               'Model 3']

################   DATA OUTPUT (Saving in Excel)    ###############
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('RESULTS_'+sysName+'_'+testType+'_'+str(numEpochs)+'Epochs.xlsx', engine='xlsxwriter') #CHANGE THE NAME OF THE OUTPUT EXCEL FILE HERE

Results = pd.DataFrame({'Model': names, 'Low': results_low,'Mid': results_mid, 'High': results_high})

# Convert the dataframe to an XlsxWriter Excel object.
Results.to_excel(writer, sheet_name=sysName)

# Close the Pandas Excel writer and output the Excel file.
writer.save()

print("PROGRAM IS COMPLETE !!!!! ")