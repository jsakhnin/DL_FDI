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
        
    
#################################################################################
sysName = "IEEE57"
sysName2 = "IEEE57_2"
testType = "ClassifiersTest_57_2"
X = []
Xv = []
Xt = []
y = []
yv = []
yt = []

for i in range(1,11,1):
    X.append(pd.read_csv("Data/{}/{}Data_10k_{}Sparsity.csv".format(sysName2, sysName, i/10 ), header=None))
    Xt.append(pd.read_csv("Data/{}/{}Data_2k_{}Sparsity.csv".format(sysName2, sysName, i/10 ), header=None))
    y.append(pd.read_csv("Data/{}/{}Labels_10k_{}Sparsity.csv".format(sysName2, sysName, i/10 ), header=None))
    yt.append(pd.read_csv("Data/{}/{}Labels_2k_{}Sparsity.csv".format(sysName2, sysName, i/10 ), header=None))
    
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
svmL = svm.SVC(kernel = 'linear')
svmR = svm.SVC(kernel = 'rbf')

print("Initializing training")
#Training Classifiers
time1= time.time()
dtc.fit(X_train,y_train)
time2 = time.time()
dctTime = time2 - time1;
print("DT DONE")

result_dct, f1_dct, precision_dct, recall_dct, fp_dct = evaluateClassifier(dtc, Xt,yt)


time1 = time.time()
knn.fit(X_train,y_train)
time2 = time.time()
knnTime = time2 - time1;
print("KNN DONE")
result_knn, f1_knn, precision_knn, recall_knn, fp_knn = evaluateClassifier(knn, Xt,yt)

time1 = time.time()
gnb.fit(X_train,y_train)
time2 = time.time()
gnbTime = time2 - time1;
print("GNB DONE")
result_gnb, f1_gnb, precision_gnb, recall_gnb, fp_gnb = evaluateClassifier(gnb, Xt,yt)

time1 = time.time()
svmL.fit(X_train,y_train)
time2 = time.time()
svmLTime = time2 - time1;
print("Linear SVM done")
result_svml, f1_svml, precision_svml, recall_svml, fp_svml = evaluateClassifier(svmL, Xt,yt)

time1 = time.time()
svmR.fit(X_train,y_train)
time2 = time.time()
svmRTime = time2 - time1;
print("RBF SVM done")

result_svmr, f1_svmr, precision_svmr, recall_svmr, fp_svmr = evaluateClassifier(svmR, Xt,yt)
    
finalTimes = [dctTime, 
              knnTime, 
              gnbTime, 
              svmLTime,
              svmRTime]

finalModels = ['DCT',
               'KNN',
               'NB',
               'SVM-L',
               'SVM-R'
               ]

sparsity = np.arange(0.1,1.1,0.1)

################   DATA OUTPUT (Saving in Excel)    ###############
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('RESULTS_'+sysName+'_'+testType+'.xlsx', engine='xlsxwriter') #CHANGE THE NAME OF THE OUTPUT EXCEL FILE HERE


Results_dct = pd.DataFrame({'Sparsity': sparsity, 'Accuracy': result_dct, 'F1 score': f1_dct, 'Precision': precision_dct, 'Recall': recall_dct, 'False Positive Rate': fp_dct})
Results_gnb = pd.DataFrame({'Sparsity': sparsity, 'Accuracy': result_gnb, 'F1 score': f1_gnb, 'Precision': precision_gnb, 'Recall': recall_gnb, 'False Positive Rate': fp_gnb})
Results_knn = pd.DataFrame({'Sparsity': sparsity, 'Accuracy': result_knn, 'F1 score': f1_knn, 'Precision': precision_knn, 'Recall': recall_knn, 'False Positive Rate': fp_knn})
Results_svml = pd.DataFrame({'Sparsity': sparsity, 'Accuracy': result_svml, 'F1 score': f1_svml, 'Precision': precision_svml, 'Recall': recall_svml, 'False Positive Rate': fp_svml})
Results_svmr = pd.DataFrame({'Sparsity': sparsity, 'Accuracy': result_svmr, 'F1 score': f1_svmr, 'Precision': precision_svmr, 'Recall': recall_svmr, 'False Positive Rate': fp_svmr})

Results_times = pd.DataFrame({'Model': finalModels, 'Training Time': finalTimes})

# Convert the dataframe to an XlsxWriter Excel object.
Results_dct.to_excel(writer, sheet_name="DCT")
Results_gnb.to_excel(writer, sheet_name="GNB")
Results_knn.to_excel(writer, sheet_name="KNN")
Results_svml.to_excel(writer, sheet_name="SVM_L")
Results_svmr.to_excel(writer, sheet_name="SVM-R")

Results_times.to_excel(writer, sheet_name="Training Time")

# Close the Pandas Excel writer and output the Excel file.
writer.save()

print("PROGRAM IS COMPLETE !!!!! ")