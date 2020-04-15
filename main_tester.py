import sys
import time
import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

###################################################################################
#####################            Functions             ############################
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

def evaluateModel(model,Xt,yt):
    accuracy = []
    f1 = []
    precision = []
    recall = []
    falsePositives = []
    
    for i in range(10):
        resultTemp = model.evaluate(Xt[i], yt[i]) #Acc
        predtemp = np.rint(model.predict(Xt[i]) )
        
        f1temp = f1_score(yt[i], predtemp) #F1
        prectemp = precision_score(yt[i], predtemp) #Precision
        recalltemp = recall_score(yt[i], predtemp) #Recall
        
        accuracy.append(resultTemp[1])
        f1.append(f1temp)
        precision.append(prectemp)
        recall.append(recalltemp)
        
        CM = confusion_matrix(yt[i], predtemp)
        
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        
        fprateTemp = FP/(TN+FP)
        falsePositives.append(fprateTemp)
        
    return accuracy, f1, precision, recall, falsePositives

def dataProc(X1,y1):
    X1 = pd.DataFrame(X1)
    X1=X1.to_numpy()
    y1 = y1.to_numpy()


    min_max_scaler = preprocessing.MinMaxScaler()
    X1 = min_max_scaler.fit_transform(X1)

    X1, y1 = shuffle(X1, y1, random_state=0)
    return X1,y1

###########################################################################################
##############            Loading Data and Models             #############################
sysName = "IEEE57"
testType = "MainModelsTest_earlyStop"
numEpochs = 100


Xt = []
yt = []

for i in range(1,11,1):
    print (i)
    Xt.append(pd.read_csv("Data/{}/{}Data_2k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None))
    yt.append(pd.read_csv("Data/{}/{}Labels_2k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None))

    Xt[i-1],yt[i-1] = dataProc(Xt[i-1],yt[i-1])


checkpoint_path = "Saved Models/"+sysName+"_models/"
numFeatures = len(Xt[0][0])

X_test = np.concatenate((Xt[0],Xt[1]), axis=0)
y_test = np.concatenate((yt[0],yt[1]), axis=0)

for i in range(2,10,1):
    X_test = np.concatenate((X_test,Xt[i]), axis=0) 
    y_test = np.concatenate((y_test,yt[i]), axis=0)

X_test, y_test = shuffle(X_test, y_test, random_state=0)

print(numFeatures)



model1 = m.DLmodel1(numFeatures)
#model2 = m.DLmodel2(numFeatures)
#model3 = m.DLmodel3(numFeatures)
model7 = m.DLmodel6(numFeatures)

model1.load_weights(checkpoint_path+'model1_100.h5')
#model2.load_weights(checkpoint_path+'model2100.h5')
#model3.load_weights(checkpoint_path+'model3.h5')
model7.load_weights(checkpoint_path+'model7_EarlyStop.h5')

###########################         Evaluating         #############################
result1, f1_1, precision1, recall1, fp1 = evaluateModel(model1, Xt,yt)
result7, f1_7, precision7, recall7, fp7 = evaluateModel(model7, Xt,yt)
sparsity = np.arange(0.1,1.1,0.1)

################   DATA OUTPUT (Saving in Excel)    ###############
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('output/RESULTS_'+sysName+'_'+testType+'_'+str(numEpochs)+'Epochs.xlsx', engine='xlsxwriter') #CHANGE THE NAME OF THE OUTPUT EXCEL FILE HERE

Results = pd.DataFrame({'Sparsity': sparsity, 'Model 1 Accuracy': result1, 'Model 7 Accuracy': result7,
                        'Model 1 F1': f1_1, 'Model 7 F1': f1_7,
                        'Model 1 Precision': precision1, 'Model 7 Precision': precision7,
                        'Model 1 Recall': recall1, 'Model 7 Recall': recall7,
                        'Model 1 False Positive Rate': fp1, 'Model 7 False Positive Rate': fp7})

# Convert the dataframe to an XlsxWriter Excel object.
Results.to_excel(writer, sheet_name=sysName)

# Close the Pandas Excel writer and output the Excel file.
writer.save()

print("PROGRAM IS COMPLETE !!!!! ")
