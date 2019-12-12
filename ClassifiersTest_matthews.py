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
from sklearn.utils import shuffle
from sklearn import tree
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import matthews_corrcoef
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


def evaluateMAT (clf, Xt, yt):
    y_pred = clf.predict(Xt)
    m = matthews_corrcoef(yt,y_pred)
    
    return m




    
#################################################################################

def classifierTest(sysName):
    testType = "ClassifiersTest_Matthews"
    X = []
    Xv = []
    Xt = []
    y = []
    yv = []
    yt = []
    
    for i in range(1,11,1):
        X.append(pd.read_csv("Data/{}/{}Data_10k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None))
        Xt.append(pd.read_csv("Data/{}/{}Data_2k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None))
        y.append(pd.read_csv("Data/{}/{}Labels_10k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None))
        yt.append(pd.read_csv("Data/{}/{}Labels_2k_{}Sparsity.csv".format(sysName, sysName, i/10 ), header=None))
        
        X[i-1],y[i-1] = dataProc(X[i-1],y[i-1])
        Xt[i-1],yt[i-1] = dataProc(Xt[i-1],yt[i-1])
        
    
    
    numFeatures = len(X[0])
    X_train = np.concatenate((X[0],X[1]), axis=0)
    X_test = np.concatenate((Xt[0],Xt[1]), axis=0)
    
    y_train = np.concatenate((y[0],y[1]), axis=0)
    y_test = np.concatenate((yt[0],yt[1]), axis=0)
    
    for i in range(2,10,1):
        X_train = np.concatenate((X_train,X[i]), axis=0) 
        X_test = np.concatenate((X_test,Xt[i]), axis=0) 
        y_train = np.concatenate((y_train,y[i]), axis=0)
        y_test = np.concatenate((y_test,yt[i]), axis=0)
    
    
    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    X_test, y_test = shuffle(X_test, y_test, random_state=0)
    
    #######################################################################################
    #Classifiers
    gnb = GaussianNB()
    knn = KNeighborsClassifier(n_neighbors=20)
    dtc = tree.DecisionTreeClassifier()

    
    print("Initializing training")
    #Training Classifiers
    time1 = time.time()
    gnb.fit(X_train,y_train)
    time2 = time.time()
    gnbTime = time2 - time1;
    print("GNB DONE")
    m_gnb = evaluateMAT(gnb, X_test, y_test)
    
    time1= time.time()
    dtc.fit(X_train,y_train)
    time2 = time.time()
    dctTime = time2 - time1;
    print("DT DONE")
    
    m_dtc = evaluateMAT(dtc, X_test, y_test)    
    
    time1 = time.time()
    knn.fit(X_train,y_train)
    time2 = time.time()
    knnTime = time2 - time1;
    print("KNN DONE")
    m_knn = evaluateMAT(knn, X_test, y_test)    
    print("Evaluation Done")
    
    finalModels = ['DTC',
                   'KNN',
                   'NB'
                   ]
    
    MAT = [m_dtc,
           m_knn,
           m_gnb]
    
    ################   DATA OUTPUT (Saving in Excel)    ###############
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('RESULTS_'+sysName+'_'+testType+'.xlsx', engine='xlsxwriter') #CHANGE THE NAME OF THE OUTPUT EXCEL FILE HERE
    
    
    Results = pd.DataFrame({'Model': finalModels, 'Matthews correlation coefficient': MAT})
    
    
    #Results_times = pd.DataFrame({'Model': finalModels, 'Training Time': finalTimes})
    
    # Convert the dataframe to an XlsxWriter Excel object.
    Results.to_excel(writer, sheet_name="classifiers")
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()    












classifierTest("IEEE14")
classifierTest("IEEE30")
classifierTest("IEEE57")
print("PROGRAM IS COMPLETE !!!!! ")