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
    result = []
    
    for i in range(10):
        resultTemp = model.evaluate(Xt[i], yt[i])
        print(resultTemp)
        result.append(resultTemp[1])
    return result

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
sysName = "IEEE30"
testType = "MainModelsTest"
numEpochs = 10


Xt = []
yt = []

for i in range(1,11,1):
    print (i)
    Xt.append(pd.read_csv("Data/{}/{}Data_2k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None))
    yt.append(pd.read_csv("Data/{}/{}Labels_2k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None))

    Xt[i-1],yt[i-1] = dataProc(Xt[i-1],yt[i-1])

checkpoint_path = "Saved Models/"+sysName+"_models/"
numFeatures = len(Xt[0][0])
print(numFeatures)
#model1 = m.DLmodel1(numFeatures)
model2 = m.DLmodel2(numFeatures)
#model3 = m.DLmodel3(numFeatures)


#model1.load_weights(checkpoint_path+'model1.h5')
model2.load_weights(checkpoint_path+'model2100.h5')
#model3.load_weights(checkpoint_path+'model3.h5')
###########################         Evaluating         #############################
results1 = evaluateModel(model2, Xt,yt)

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
