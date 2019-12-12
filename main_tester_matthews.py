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
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import matthews_corrcoef

###################################################################################
#####################            Functions             ############################
def evaluateMAT (clf, Xt, yt):
    y_pred = clf.predict(Xt)
    y_pred = np.rint(y_pred)
    m = matthews_corrcoef(yt,y_pred)
    
    return m

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
def modelMAT(sysName):
    testType = "MainModelsTest_Matthews"
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
    model7 = m.DLmodel6(numFeatures)
    
    model1.load_weights(checkpoint_path+'model1_100.h5')
    model7.load_weights(checkpoint_path+'model7_EarlyStop.h5')
    
    ###########################         Evaluating         #############################
    m1 = evaluateMAT(model1, X_test,y_test)
    m7 = evaluateMAT(model7, X_test,y_test)
    
    models = ['ANN', 'GDNN']
    MAT = [m1, m7]
    ################   DATA OUTPUT (Saving in Excel)    ###############
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('RESULTS_'+sysName+'_'+testType+'.xlsx', engine='xlsxwriter') #CHANGE THE NAME OF THE OUTPUT EXCEL FILE HERE
    
    
    Results = pd.DataFrame({'Models': models, 'Matthew Correlation Coefficient': MAT})
    
    
    #Results_times = pd.DataFrame({'Model': finalModels, 'Training Time': finalTimes})
    
    # Convert the dataframe to an XlsxWriter Excel object.
    Results.to_excel(writer, sheet_name="Models")
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()  


modelMAT("IEEE14")
modelMAT("IEEE30")
modelMAT("IEEE57")
print("PROGRAM IS COMPLETE !!")