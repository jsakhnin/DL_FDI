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


def evaluateClassifier (clf, Xt, yt):
    accuracy = []
    f1 = []
    precision = []
    recall = []
    falsePositives = []
    for i in range(10):
        y_pred = clf.predict(Xt[i])
        CM = confusion_matrix(yt[i], y_pred)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        
        accuracy.append( ((TP+TN)/(TP+FN+TN+FP))  )
        precision.append(  (TP/(TP+FP)) )
        recall.append(  (TP/(TP+FN))  )
        f1.append ( ( (2*precision[i]*recall[i])/(precision[i]+recall[i]) )  )
        falsePositives.append( (FP/(TN+FP))  )
        
    return accuracy, f1, precision, recall, falsePositives

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
##########################################################################################################
#################################             Deep Learning                          #####################
def imb_test (sysName):
    testType = "ClassifiersTest_Imbalanced"    
    imbalanceRange = np.arange(0.1,1.0,0.1)
    Results_dct = []
    Results_knn = []
    Results_gnb = []
    
    #Classifiers
    gnb = GaussianNB()
    knn = KNeighborsClassifier(n_neighbors=20)
    dtc = tree.DecisionTreeClassifier()
    
    Xt, yt = load_testData(sysName)
    sparsity = np.arange(0.1,1.1,0.1)
    
    for imbalance in imbalanceRange:
        checkpoint_path = "Saved Models/"+sysName+"_models/{}".format(imbalance)
        X_train, X_val, y_train, y_val = imbalance_data(sysName, imbalance)
        numFeatures = len(X_train[0])
        print("Features = ", numFeatures)   
        dtc.fit(X_train,y_train)
        result_dct, f1_dct, precision_dct, recall_dct, fp_dct = evaluateClassifier(dtc, Xt,yt)    
        knn.fit(X_train,y_train)
        result_knn, f1_knn, precision_knn, recall_knn, fp_knn = evaluateClassifier(knn, Xt,yt)    
        gnb.fit(X_train,y_train)
        result_gnb, f1_gnb, precision_gnb, recall_gnb, fp_gnb = evaluateClassifier(gnb, Xt,yt)
        
        Results_dct.append(  pd.DataFrame({'Sparsity': sparsity, 'Accuracy': result_dct, 'F1 score': f1_dct, 'Precision': precision_dct, 'Recall': recall_dct, 'False Positive Rate': fp_dct})  )
        Results_gnb.append(  pd.DataFrame({'Sparsity': sparsity, 'Accuracy': result_gnb, 'F1 score': f1_gnb, 'Precision': precision_gnb, 'Recall': recall_gnb, 'False Positive Rate': fp_gnb})  )
        Results_knn.append(  pd.DataFrame({'Sparsity': sparsity, 'Accuracy': result_knn, 'F1 score': f1_knn, 'Precision': precision_knn, 'Recall': recall_knn, 'False Positive Rate': fp_knn})  )
        
    
    ################   DATA OUTPUT (Saving in Excel)    ###############
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer_dct = pd.ExcelWriter('RESULTS_'+sysName+'_'+testType+'_DCT.xlsx', engine='xlsxwriter') #CHANGE THE NAME OF THE OUTPUT EXCEL FILE HERE
    writer_gnb = pd.ExcelWriter('RESULTS_'+sysName+'_'+testType+'_GNB.xlsx', engine='xlsxwriter') #CHANGE THE NAME OF THE OUTPUT EXCEL FILE HERE
    writer_knn = pd.ExcelWriter('RESULTS_'+sysName+'_'+testType+'_KNN.xlsx', engine='xlsxwriter') #CHANGE THE NAME OF THE OUTPUT EXCEL FILE HERE
    
    for i in range(9):
        Results_dct[i].to_excel(writer_dct, sheet_name = "Imb {}".format(imbalanceRange[i]) )
        Results_gnb[i].to_excel(writer_gnb, sheet_name = "Imb {}".format(imbalanceRange[i]) )
        Results_knn[i].to_excel(writer_knn, sheet_name = "Imb {}".format(imbalanceRange[i]) )
        
    # Close the Pandas Excel writer and output the Excel file.
    writer_dct.save()
    writer_gnb.save()
    writer_knn.save()






imb_test("IEEE30")
imb_test("IEEE14")
imb_test("IEEE57")
print("PROGRAM IS COMPLETE !!!!! ")