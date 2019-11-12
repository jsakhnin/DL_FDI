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

#################################################################################
sysName = "IEEE14"
testType = "ClassifiersTest_Imbalanced"

X1 = pd.read_csv("Data/"+sysName+"/"+sysName+"Data_40k_lowSparsity"+".csv",header=None).iloc[:25000]
y1 = pd.read_csv("Data/"+sysName+"/"+sysName+"Labels_40k_lowSparsity"+".csv",header=None).iloc[:25000]
X2 = pd.read_csv("Data/"+sysName+"/"+sysName+"Data_40k_midSparsity"+".csv",header=None).iloc[:25000]
y2 = pd.read_csv("Data/"+sysName+"/"+sysName+"Labels_40k_midSparsity"+".csv",header=None).iloc[:25000]
X3 = pd.read_csv("Data/"+sysName+"/"+sysName+"Data_40k_highSparsity"+".csv",header=None).iloc[:25000]
y3 = pd.read_csv("Data/"+sysName+"/"+sysName+"Labels_40k_highSparsity"+".csv",header=None).iloc[:25000]

Xv1 = pd.read_csv("Data/"+sysName+"/"+sysName+"Data_10k_lowSparsity"+".csv",header=None).iloc[:6250]
yv1 = pd.read_csv("Data/"+sysName+"/"+sysName+"Labels_10k_lowSparsity"+".csv",header=None).iloc[:6250]
Xv2 = pd.read_csv("Data/"+sysName+"/"+sysName+"Data_10k_midSparsity"+".csv",header=None).iloc[:6250]
yv2 = pd.read_csv("Data/"+sysName+"/"+sysName+"Labels_10k_midSparsity"+".csv",header=None).iloc[:6250]
Xv3 = pd.read_csv("Data/"+sysName+"/"+sysName+"Data_10k_highSparsity"+".csv",header=None).iloc[:6250]
yv3 = pd.read_csv("Data/"+sysName+"/"+sysName+"Labels_10k_highSparsity"+".csv",header=None).iloc[:6250]

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
X = np.concatenate((X,Xv), axis = 0)
y = np.concatenate((y,yv), axis = 0)

Xt = np.concatenate((Xt1, Xt2, Xt3), axis=0)
yt = np.concatenate((yt1, yt2, yt3), axis=0)

X, y = shuffle(X, y, random_state=0)
Xt, yt = shuffle(Xt, yt, random_state=0)

#######################################################################################
#Classifiers
gnb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=20)
dtc = tree.DecisionTreeClassifier()
svmL = svm.SVC(kernel = 'linear')
svmR = svm.SVC(kernel = 'rbf')

print("Initializing training")
#Training Classifiers
time1= time.time()
dtc.fit(X,y)
time2 = time.time()
dctTime = time2 - time1;
print("DT DONE")

time1 = time.time()
knn.fit(X,y)
time2 = time.time()
knnTime = time2 - time1;
print("KNN DONE")

time1 = time.time()
gnb.fit(X,y)
time2 = time.time()
gnbTime = time2 - time1;
print("GNB DONE")

time1 = time.time()
svmL.fit(X,y)
time2 = time.time()
svmLTime = time2 - time1;
print("Linear SVM done")

time1 = time.time()
svmR.fit(X,y)
time2 = time.time()
svmRTime = time2 - time1;
print("RBF SVM done")

 
##Testing classifiers
dctResult_low = dtc.score(Xt1,yt1)
dctResult_mid = dtc.score(Xt2,yt2)
dctResult_high = dtc.score(Xt3,yt3)

gnbResult_low = gnb.score(Xt1,yt1)
gnbResult_mid = gnb.score(Xt2,yt2)
gnbResult_high = gnb.score(Xt3,yt3)


knnResult_low = knn.score(Xt1,yt1)
knnResult_mid = knn.score(Xt2,yt2)
knnResult_high = knn.score(Xt3,yt3)

svmLResult_low = svmL.score(Xt1,yt1)
svmLResult_mid = svmL.score(Xt2,yt2)
svmLResult_high = svmL.score(Xt3,yt3)

svmRResult_low = svmR.score(Xt1,yt1)
svmRResult_mid = svmR.score(Xt2,yt2)
svmRResult_high = svmR.score(Xt3,yt3)

print("DCT high: ", dctResult_high)
print("DCT mid: ", dctResult_mid)
print("DCT low: ", dctResult_low)

print("GNB high: ", gnbResult_high)
print("GNB mid: ", gnbResult_mid)
print("GNB low: ", gnbResult_low)

print("KNN high: ", knnResult_high)
print("KNN mid: ", knnResult_mid)
print("KNN low: ", knnResult_low)

finalResults = [dctResult_low, dctResult_mid, dctResult_high,
                knnResult_low, knnResult_mid, knnResult_high,
                gnbResult_low, gnbResult_mid, gnbResult_high,
                svmLResult_low, svmLResult_mid, svmLResult_high,
                svmRResult_low, svmRResult_mid, svmRResult_high
                ]

finalTimes = [dctTime, dctTime, dctTime, 
              knnTime, knnTime, knnTime,
              gnbTime, gnbTime, gnbTime,
              svmLTime, svmLTime, svmLTime,
              svmRTime, svmRTime, svmRTime]

finalModels = ['DCT low', 'DCT mid', 'DCT high',
               'KNN low', 'KNN mid','KNN high',
               'NB low', 'NB mid', 'NB high',
               'SVM-L low', 'SVM-L mid', 'SVM-L high',
               'SVM-R low', 'SVM-R mid', 'SVM-R high'
               ]


################   DATA OUTPUT (Saving in Excel)    ###############
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('RESULTS_'+sysName+'_'+testType+'.xlsx', engine='xlsxwriter') #CHANGE THE NAME OF THE OUTPUT EXCEL FILE HERE

Results = pd.DataFrame({'Model': finalModels, 'Accuracy': finalResults, 'Training Time': finalTimes})

# Convert the dataframe to an XlsxWriter Excel object.
Results.to_excel(writer, sheet_name=sysName)

# Close the Pandas Excel writer and output the Excel file.
writer.save()

print("PROGRAM IS COMPLETE !!!!! ")