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
###################################################################################
#####################            Functions             ############################
def evaluateROC (clf, Xt, yt):
    aucScores = []
    y_pred = clf.predict(Xt)
    aucScore = roc_auc_score(yt, y_pred)    
    fpr, tpr, thresholds = metrics.roc_curve(yt,y_pred)
    
    return fpr, tpr, thresholds, aucScore

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
def modelAUC(sysName):
    testType = "MainModelsTest_AUC"
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
    fpr1, tpr1, thresh1, auc1 = evaluateROC(model1, X_test,y_test)
    fpr7, tpr7, thresh7, auc7 = evaluateROC(model7, X_test,y_test)
    ################   DATA OUTPUT (Saving in Excel)    ###############
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('output/RESULTS_'+sysName+'_'+testType+'.xlsx', engine='xlsxwriter') #CHANGE THE NAME OF THE OUTPUT EXCEL FILE HERE
    
    
    Results1 = pd.DataFrame({'False Positive Rate': fpr1, 'True Positive Rate': tpr1, 'Thresholds': thresh1, 'AUC': auc1})
    Results7 = pd.DataFrame({'False Positive Rate': fpr7, 'True Positive Rate': tpr7, 'Thresholds': thresh7, 'AUC': auc7})
    
    
    #Results_times = pd.DataFrame({'Model': finalModels, 'Training Time': finalTimes})
    
    # Convert the dataframe to an XlsxWriter Excel object.
    Results1.to_excel(writer, sheet_name="Model 1")
    Results7.to_excel(writer, sheet_name="Model 7")
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()  


modelAUC("IEEE14")
modelAUC("IEEE30")
modelAUC("IEEE57")
print("PROGRAM IS COMPLETE !!")
