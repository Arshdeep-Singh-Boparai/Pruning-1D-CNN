#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 16:55:38 2018

@author: arshdeep
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:42:02 2017

@author: arshdeep
"""
import os
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import svm
from scipy.stats import mode
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
#%%

os.chdir('/home/arshdeep/DCASE2016/numpy/FOLD1')
print("Data Read")
labels=np.load('labels_train.npy')
labels_test=np.load('labels_test.npy')
dirs=np.load('dire_list.npy')
ex4=np.load('ex4.npy')
ex_test4=np.load('ex_test4.npy')

ex6=np.load('ex6.npy')
ex_test6=np.load('ex_test6.npy')

ex7=np.load('ex7.npy')
ex_test7=np.load('ex_test7.npy')


ex8=np.load('ex8.npy')
ex_test8=np.load('ex_test8.npy')

ex10=np.load('ex10.npy')
ex_test10=np.load('ex_test10.npy')

ex11=np.load('ex11.npy')
ex_test11=np.load('ex_test11.npy')
ex13=np.load('ex13.npy')
ex_test13=np.load('ex_test13.npy')


ex14=np.load('ex14.npy')
ex_test14=np.load('ex_test14.npy')

ex16=np.load('ex16.npy')
ex_test16=np.load('ex_test16.npy')

ex17=np.load('ex17.npy')
ex_test17=np.load('ex_test17.npy')

ex18=np.load('ex18.npy')
ex_test18=np.load('ex_test18.npy')


ex20=np.load('ex20.npy')
ex_test20=np.load('ex_test20.npy')

ex21=np.load('ex21.npy')
ex_test21=np.load('ex_test21.npy')

ex23=np.load('ex23.npy')
ex_test23=np.load('ex_test23.npy')

ex24=np.load('ex24.npy')
ex_test24=np.load('ex_test24.npy')
#%%
'''
ex_test_challenge_4=np.load('ex_test_chal4.npy')
ex_test_challenge_6=np.load('ex_test_chal6.npy')
ex_test_challenge_7=np.load('ex_test_chal7.npy')
ex_test_challenge_8=np.load('ex_test_chal8.npy')
ex_test_challenge_10=np.load('ex_test_chal10.npy')
ex_test_challenge_11=np.load('ex_test_chal11.npy')
ex_test_challenge_13=np.load('ex_test_chal13.npy')
ex_test_challenge_14=np.load('ex_test_chal14.npy')
ex_test_challenge_16=np.load('ex_test_chal16.npy')
ex_test_challenge_17=np.load('ex_test_chal17.npy')
ex_test_challenge_18=np.load('ex_test_chal18.npy')
ex_test_challenge_20=np.load('ex_test_chal20.npy')
ex_test_challenge_21=np.load('ex_test_chal21.npy')
ex_test_challenge_23=np.load('ex_test_chal23.npy')
ex_test_challenge_24=np.load('ex_test_chal24.npy')

labels_test_chal=np.load('labels_test_chal.npy')
'''
#%%


labels=np.array(labels)
labels_test=np.asanyarray(labels_test)
print("SVM training started......")
target_names = dirs#['library', 'metro', 'city','car','forest','residen','beach','train','bus','office','park','cafe','tram','home','grocer']

y=labels
#y=np.append(y,labels_test)
X23=[]
clf23=[]
#ex_23=np.vstack((ex23,ex_test23))

X23=np.array(normalize(np.array(ex23), norm='l2', axis=1, copy=True, return_norm=False))

clf23 = svm.SVC(kernel='poly',degree=7,gamma=3,decision_function_shape = "ovo",probability=True,random_state=0)
#clf.decision_function_shape = "ovr"
clf23.fit(X23, y)
prob23_pred=[]
#sum23_pred=clf23.predict(normalize(ex_test23, norm='l2', axis=1, copy=True, return_norm=False))
prob23_pred=clf23.predict_proba(normalize(np.array(ex_test23), norm='l2', axis=1, copy=True, return_norm=False))
#



predd=np.argmax(prob23_pred,1)
#
#print(classification_report(labels_test,predd,target_names=target_names))
##
##
#asd=confusion_matrix(labels_test,predd);
#accu=(np.trace(asd)/289)*100;
#print(accu)
##
#target_names = dirs#['library', 'metro', 'city','car','forest','residen','beach','train','bus','office','park','cafe','tram','home','grocer']
#print(classification_report(labels_test,predd,target_names=dirs))
asd=confusion_matrix(labels_test,predd);
accu=(np.trace(asd)/np.size(labels_test))*100;
print("accuracy layer 23......")
print(accu)
#%%
X24=[]
clf24=[]
#ex_24=np.vstack((ex24,ex_test24))

X24=np.array(normalize(ex24, norm='l2', axis=1, copy=True, return_norm=False))
#y=np.array(labels)
clf24 = svm.SVC(kernel='poly',degree=5,gamma=3,decision_function_shape = "ovo",probability=True,random_state=0)
#clf.decision_function_shape = "ovr"
clf24.fit(X24, y)
#sum24_pred=clf24.predict(normalize(ex_test24, norm='l2', axis=1, copy=True, return_norm=False))
#prob24_pred=[]
prob24_pred=clf24.predict_proba(normalize(ex_test24, norm='l2', axis=1, copy=True, return_norm=False))
predd=[]
predd=np.argmax(prob24_pred,1)
##

###
###
#asd=confusion_matrix(labels_test,sum24_pred);
#accu=(np.trace(asd)/289)*100;
#print(accu)
#
#
#target_names = ['library', 'metro', 'city','car','forest','residen','beach','train','bus','office','park','cafe','tram','home','grocer']
#print(classification_report(labels_test,sum24_pred,target_names=target_names))
asd=confusion_matrix(labels_test,predd);
accu=(np.trace(asd)/np.size(labels_test))*100;
print("accuracy layer 24......")
print(accu)
#print(classification_report(labels_test,predd,target_names=target_names))


#%%




#%%sum pool
X21=[]
clf21=[]
prob21_pred=[]
X21=np.array(normalize(ex21, norm='l2', axis=1, copy=True, return_norm=False))
#y=np.array(labels)
clf21 = svm.SVC(kernel='poly',degree=3,gamma=10,decision_function_shape = "ovo",probability=True,random_state=0)
#clf.decision_function_shape = "ovr"
clf21.fit(X21, y)
#sum21_pred=clf21.predict(normalize(ex_test21, norm='l2', axis=1, copy=True, return_norm=False))
prob21_pred=clf21.predict_proba(normalize(ex_test21, norm='l2', axis=1, copy=True, return_norm=False))
##
predd=[]
predd=np.argmax(prob21_pred,1)
##

###
###
#asd=confusion_matrix(labels_test,predd);
#accu=(np.trace(asd)/298)*100;
#print(accu)
##
#target_names = ['library', 'metro', 'city','car','forest','residen','beach','train','bus','office','park','cafe','tram','home','grocer']
#print(classification_report(labels_test,sum21_pred,target_names=target_names))
asd=confusion_matrix(labels_test,predd);
accu=(np.trace(asd)/np.size(labels_test))*100;
print("accuracy layer 21......")
print(accu)
#print(classification_report(labels_test,predd,target_names=target_names))
#%%
X20=[]
clf20=[]
prob20_pred=[]
#ex_20=np.vstack((ex20,ex_test20))
X20=np.array(normalize(ex20, norm='l2', axis=1, copy=True, return_norm=False))
#y=np.array(labels)
clf20 = svm.SVC(kernel='poly',degree=18,gamma=50,decision_function_shape = "ovo",probability=True,random_state=0)
#clf.decision_function_shape = "ovr"
clf20.fit(X20, y)
#sum20_pred=clf20.predict(normalize(np.array(ex_test20), norm='l2', axis=1, copy=True, return_norm=False))
prob20_pred=clf20.predict_proba(normalize(ex_test20, norm='l2', axis=1, copy=True, return_norm=False))
##
predd=[]
predd=np.argmax(prob20_pred,1)
#

##
##
#asd=confusion_matrix(labels_test,predd);
#accu=(np.trace(asd)/298)*100;
#print(accu)
##
##
#target_names = ['library', 'metro', 'city','car','forest','residen','beach','train','bus','office','park','cafe','tram','home','grocer']
#print(classification_report(labels_test,sum20_pred,target_names=target_names))
asd=confusion_matrix(labels_test,predd);
accu=(np.trace(asd)/np.size(labels_test))*100;
print("accuracy layer 20......")
print(accu)
#print(classification_report(labels_test,predd,target_names=target_names))
#%%
#ex_14=np.vstack((ex14,ex_test14))
x14=[]
clf14=[]
prob14_pred=[]


X14=np.array(normalize((np.array(ex14)), norm='l2', axis=1, copy=True, return_norm=False))
#y=np.array(labels)
clf14 = svm.SVC(kernel='poly',degree=3,gamma=50,decision_function_shape = "ovo",probability=True,random_state=0)
#clf.decision_function_shape = "ovr"
clf14.fit(X14, y)
#sum14_pred=clf14.predict(normalize(np.array(ex_test14), norm='l2', axis=1, copy=True, return_norm=False))
prob14_pred=clf14.predict_proba(normalize((np.array(ex_test14)), norm='l2', axis=1, copy=True, return_norm=False))
predd=[]
predd=np.argmax(prob14_pred,1)
#
#print(classification_report(labels_test,predd,target_names=target_names))
##
##
#asd=confusion_matrix(labels_test,predd);y=labels

#accu=(np.trace(asd)/298)*100;
#print(accu)
##
#
#
#target_names = ['library', 'metro', 'city','car','forest','residen','beach','train','bus','office','park','cafe','tram','home','grocer']

asd=confusion_matrix(labels_test,predd);
accu=(np.trace(asd)/np.size(labels_test))*100;
print("accuracy layer 14......")
print(accu)
#print(classification_report(labels_test,predd,target_names=dirs))
#%%

X17=[]
clf17=[]
prob17_pred=[]

#ex_17=np.vstack((ex17,ex_test17))
X17=np.array(normalize(ex17, norm='l2', axis=1, copy=True, return_norm=False))
#y=np.array(labels)
clf17 = svm.SVC(kernel='poly',degree=4,gamma=20,decision_function_shape = "ovo",probability=True,random_state=0)
#clf.decision_function_shape = "ovr"
clf17.fit(X17, y)
#sum17_pred=clf17.predict(normalize(ex_test17, norm='l2', axis=1, copy=True, return_norm=False))
prob17_pred=clf17.predict_proba(normalize(ex_test17, norm='l2', axis=1, copy=True, return_norm=False))
#
predd=[]
predd=np.argmax(prob17_pred,1)


##
##
#asd=confusion_matrix(labels_test,predd);
#accu=(np.trace(asd)/298)*100;
#print(accu)
##
#
##target_names = ['library', 'metro', 'city','car','forest','residen','beach','train','bus','office','park','cafe','tram','home','grocer']
##print(classification_report(labels_test,sum17_pred,target_names=target_names))
##
##
#asd=confusion_matrix(labels_test,sum17_pred);
#accu=(np.trace(asd)/298)*100;
#print(accu)
#
asd=confusion_matrix(labels_test,predd);
accu=(np.trace(asd)/np.size(labels_test))*100;
print("accuracy layer 17......")
print(accu)
#print(classification_report(labels_test,predd,target_names=dirs))
#%% 
X18=[]
clf18=[]
prob18_pred=[]
#ex_18=np.vstack((ex18,ex_test18))
X18=np.array(normalize(ex18, norm='l2', axis=1, copy=True, return_norm=False))
#y=np.array(labels)
clf18 = svm.SVC(kernel='poly',degree=8,gamma=20,decision_function_shape = "ovo",probability=True,random_state=0)
#clf.decision_function_shape = "ovr"
clf18.fit(X18, y)

#sum18_pred=clf18.predict(normalize(ex_test18, norm='l2', axis=1, copy=True, return_norm=False))
prob18_pred=clf18.predict_proba(normalize(ex_test18, norm='l2', axis=1, copy=True, return_norm=False))
##
predd=[]
predd=np.argmax(prob18_pred,1)
#

##
##
#asd=confusion_matrix(labels_test,predd);
#accu=(np.trace(asd)/298)*100;
#print(accu)
##
##
#
##target_names = ['library', 'metro', 'city','car','forest','residen','beach','train','bus','office','park','cafe','tram','home','grocer']
#print(classification_report(labels_test,sum18_pred,target_names=target_names))
##
##
#asd=confusion_matrix(labels_test,sum18_pred);
#accu=(np.trace(asd)/298)*100;
#print(accu)

asd=confusion_matrix(labels_test,predd);
accu=(np.trace(asd)/np.size(labels_test))*100;
print("accuracy layer 18......")
print(accu)
#print(classification_report(labels_test,predd,target_names=dirs))
#%% laywer 10

X10=[]
clf10=[]
prob10_pred=[]
X10=np.array(normalize(np.array(ex10), norm='l2', axis=1, copy=True, return_norm=False))
#y=np.array(labels)
clf10 = svm.SVC(kernel='poly',degree=5,gamma=100,decision_function_shape = "ovo",probability=True,random_state=0)
#clf.decision_function_shape = "ovr"
clf10.fit(X10, y)

#sum10_pred=clf10.predict(normalize(ex_test10, norm='l2', axis=1, copy=True, return_norm=False))
prob10_pred=clf10.predict_proba(normalize(np.array(ex_test10), norm='l2', axis=1, copy=True, return_norm=False))
##
predd=[]
predd=np.argmax(prob10_pred,1)
#

##
##
#asd=confusion_matrix(labels_test,predd);
#accu=(np.trace(asd)/298)*100;
#print(accu)

#
##print(classification_report(labels_test,sum10_pred,target_names=target_names))
##
##
#asd=confusion_matrix(labels_test,sum10_pred);
#accu=(np.trace(asd)/298)*100;
#print(accu)

asd=confusion_matrix(labels_test,predd);
accu=(np.trace(asd)/np.size(labels_test))*100;
print("accuracy layer 10......")
print(accu)
#print(classification_report(labels_test,predd,target_names=dirs))
#%%
#ex_11=np.vstack((ex11,ex_test11))
X11=[]
clf11=[]
prob11_pred=[]
X11=np.array(normalize(ex11, norm='l2', axis=1, copy=True, return_norm=False))
#y=np.array(labels)
clf11 = svm.SVC(kernel='poly',degree=6,gamma=10,decision_function_shape = "ovo",probability=True,random_state=0)
#clf.decision_function_shape = "ovr"
clf11.fit(X11, y)

#sum11_pred=clf11.predict(normalize(ex_test11, norm='l2', axis=1, copy=True, return_norm=False))
prob11_pred=clf11.predict_proba(normalize(ex_test11, norm='l2', axis=1, copy=True, return_norm=False))
##
predd=[]
predd=np.argmax(prob11_pred,1)
#

##
##
#asd=confusion_matrix(labels_test,predd);
#accu=(np.trace(asd)/298)*100;
#print(accu)
##
#
##print(classification_report(labels_test,sum11_pred,target_names=target_names))
##
##
#asd=confusion_matrix(labels_test,sum11_pred);
#accu=(np.trace(asd)/298)*100;
#print(accu)
asd=confusion_matrix(labels_test,predd);
accu=(np.trace(asd)/np.size(labels_test))*100;
print("accuracy layer 11......")
print(accu)
#print(classification_report(labels_test,predd,target_names=dirs))
#%%


X13=[]
clf13=[]
prob13_pred=[]
X13=np.array(normalize(ex13, norm='l2', axis=1, copy=True, return_norm=False))
#y=np.array(labels)
clf13 = svm.SVC(kernel='poly',degree=4,gamma=40,decision_function_shape = "ovo",probability=True,random_state=0)
#clf.decision_function_shape = "ovr"
clf13.fit(X13, y)

#sum13_pred=clf13.predict(normalize(ex_test13, norm='l2', axis=1, copy=True, return_norm=False))

prob13_pred=clf13.predict_proba(normalize(ex_test13, norm='l2', axis=1, copy=True, return_norm=False))
#
predd=[]
predd=np.argmax(prob13_pred,1)
#

##
##
#asd=confusion_matrix(labels_test,predd);
#accu=(np.trace(asd)/298)*100;
#print(accu)
##
##
##print(classification_report(labels_test,sum13_pred,target_names=target_names))
##
##
#asd=confusion_matrix(labels_test,sum13_pred);
#accu=(np.trace(asd)/298)*100;
#print(accu)
#
asd=confusion_matrix(labels_test,predd);
accu=(np.trace(asd)/np.size(labels_test))*100;
print("accuracy layer 13......")
print(accu)
#print(classification_report(labels_test,predd,target_names=dirs))
#%%  layer 16

X16=[]
clf16=[]
prob16_pred=[]
X16=np.array(normalize(ex16, norm='l2', axis=1, copy=True, return_norm=False))
#y=np.array(labels)
clf16 = svm.SVC(kernel='poly',degree=10,gamma=20,decision_function_shape = "ovo",probability=True,random_state=0)
#clf.decision_function_shape = "ovr"
clf16.fit(X16, y)

#sum16_pred=clf16.predict(normalize(ex_test16, norm='l2', axis=1, copy=True, return_norm=False))
prob16_pred=clf16.predict_proba(normalize(ex_test16, norm='l2', axis=1, copy=True, return_norm=False))
#
##
predd=[]
predd=np.argmax(prob16_pred,1)
#

##
##
#asd=confusion_matrix(labels_test,predd);
#accu=(np.trace(asd)/298)*100;
#print(accu)
#
##print(classification_report(labels_test,sum16_pred,target_names=target_names))
##
##
#asd=confusion_matrix(labels_test,sum16_pred);
#accu=(np.trace(asd)/298)*100;
#print(accu)
asd=confusion_matrix(labels_test,predd);
accu=(np.trace(asd)/np.size(labels_test))*100;
print("accuracy layer 16......")
print(accu)
#print(classification_report(labels_test,predd,target_names=dirs))

#%% layer 8

X8=[]
clf8=[]
y=labels
prob8_pred=[]
X8=np.array(normalize((ex8), norm='l2', axis=1, copy=True, return_norm=False))
#y=np.array(labels)
clf8 = svm.SVC(kernel='poly', degree=7,gamma=15,decision_function_shape = "ovo",probability=True,random_state=0)
#clf.decision_function_shape = "ovr"
clf8.fit(X8, y)

#sum8_pred=clf8.predict(normalize(ex_test8, norm='l2', axis=1, copy=True, return_norm=False))
prob8_pred=clf8.predict_proba(normalize((ex_test8), norm='l2', axis=1, copy=True, return_norm=False))
#
predd=[]
predd=np.argmax(prob8_pred,1)
#

##
##
#asd=confusion_matrix(labels_test,predd);
#accu=(np.trace(asd)/298)*100;
#print(accu)
#
#
#
#
##print(classification_report(labels_test,sum8_pred,target_names=target_names))
##
##
#asd=confusion_matrix(labels_test,sum8_pred);
#accu=(np.trace(asd)/298)*100;
#print(accu)
asd=confusion_matrix(labels_test,predd);
accu=(np.trace(asd)/np.size(labels_test))*100;
print("accuracy layer 8......")
print(accu)
#print(classification_report(labels_test,predd,target_names=dirs))
#%%

X4=[]
clf4=[]
prob4_pred=[]
X4=np.array(normalize(ex4, norm='l2', axis=1, copy=True, return_norm=False))
#y=np.array(labels)
clf4 = svm.SVC(kernel='poly',degree=5,gamma=15,decision_function_shape = "ovo",probability=True,random_state=0)
#clf.decision_function_shape = "ovr"
clf4.fit(X4, y)

#sum4_pred=clf4.predict(normalize(ex_test4, norm='l2', axis=1, copy=True, return_norm=False))
prob4_pred=clf4.predict_proba(normalize(ex_test4, norm='l2', axis=1, copy=True, return_norm=False))
predd=[]
predd=np.argmax(prob4_pred,1)
#

##
##
#asd=confusion_matrix(labels_test,predd);
#accu=(np.trace(asd)/289)*100;
#print(accu)
##
#asd=confusion_matrix(labels_test,sum4_pred);
#accu=(np.trace(asd)/298)*100;
#print(accu)
asd=confusion_matrix(labels_test,predd);
accu=(np.trace(asd)/np.size(labels_test))*100;
print("accuracy layer 4......")
print(accu)
#print(classification_report(labels_test,predd,target_names=dirs))

#%%\
#ex_7=np.vstack((ex7,ex_test7))
X7=[]
clf7=[]
prob7_pred=[]
X7=np.array(normalize(ex7, norm='l2', axis=1, copy=True, return_norm=False))
#y=np.array(labels)
clf7 = svm.SVC(kernel='poly',degree=5,gamma=400,decision_function_shape = "ovo",probability=True,random_state=0)
#clf.decision_function_shape = "ovr"
clf7.fit(X7, y)

#sum7_pred=clf7.predict(normalize(np.array(ex_test7), norm='l2', axis=1, copy=True, return_norm=False))
prob7_pred=clf7.predict_proba(normalize(ex_test7, norm='l2', axis=1, copy=True, return_norm=False))
predd=[]
predd=np.argmax(prob7_pred,1)

##
#asd=confusion_matrix(labels_test,predd);
#accu=(np.trace(asd)/298)*100;
#print(accu)
##
#new_prob=np.matmul(np.diag(weight[0,:]),prob24_pred_avg)+np.matmul(np.diag(weight[1,:]),prob23_pred_avg)+np.matmul(np.diag(weight[2,:]),prob21_pred_avg)+np.matmul(np.diag(weight[3,:]),prob20_pred_avg)+np.matmul(np.diag(weight[4,:]),prob18_pred_avg)+np.matmul(np.diag(weight[5,:]),prob17_pred_avg)+np.matmul(np.diag(weight[6,:]),prob16_pred_avg)+np.matmul(np.diag(weight[7,:]),prob14_pred_avg)+np.matmul(np.diag(weight[8,:]),prob13_pred_avg)+np.matmul(np.diag(weight[9,:]),prob11_pred_avg)+np.matmul(np.diag(weight[10,:]),prob10_pred_avg)+np.matmul(np.diag(weight[11,:]),prob8_pred_avg)+np.matmul(np.diag(weight[12,:]),prob7_pred_avg)+np.matmul(np.diag(weight[13,:]),prob6_pred_avg)+np.matmul(np.diag(weight[14,:]),prob4_pred_avg)

#
#
##print(classification_report(labels_test,sum7_pred,target_names=target_names))
##
##
#asd=confusion_matrix(labels_test,sum7_pred);
#accu=(np.trace(asd)/298)*100;
#print(accu)
asd=confusion_matrix(labels_test,predd);
accu=(np.trace(asd)/np.size(labels_test))*100;
print("accuracy layer 7......")
print(accu)
#
#print(classification_report(labels_test,predd,target_names=dirs))
##
#%%

X6=[]
clf6=[]
prob6_pred=[]
X6=np.array(normalize(ex6, norm='l2', axis=1, copy=True, return_norm=False))
#y=np.array(labels)
clf6 = svm.SVC(kernel='poly',degree=5,gamma=8,decision_function_shape = "ovo",probability=True,random_state=0)
#clf.decision_function_shape = "ovr"
clf6.fit(X6, y)

#sum6_pred=clf6.predict(normalize(np.array(ex_test6), norm='l2', axis=1, copy=True, return_norm=False))
prob6_pred=clf6.predict_proba(normalize(ex_test6, norm='l2', axis=1, copy=True, return_norm=False))
#
predd=[]
predd=np.argmax(prob6_pred,1)
#

##
##
#asd=confusion_matrix(labels_test,predd);
#accu=(np.trace(asd)/289)*100;
#print(accu)
#

#print(classification_report(labels_test,sum6_pred,target_names=target_names))
#
#
#asd=confusion_matrix(labels_test,sum6_pred);4        18
#accu=(np.trace(asd)/298)*100;
##print(accu)
#
#clf6.score(normalize(np.array(ex_test6), norm='l2', axis=1, copy=True, return_norm=False), labels_test)
##%%
#X3=np.array(ex3)
#y=np.array(labels)
#clf3 = svm.SVC(kernel='linear',degree=12,gamma=0.0001,decision_function_shape = "ovo",probability=True)
##clf.decision_function_shape = "ovr"
#clf3.fit(X3, y)
#
#sum3_pred=clf3.predict(np.array(ex_test3))http://c.v.jawahar/
#
#print(classification_report(labels_test,sum3_pred,target_names=target_names))
#
##
#asd=confusion_matrix(labels_test,sum6_pred);
#accu=(np.trace(asd)/298)*100
#print(accu)
##
#
asd=confusion_matrix(labels_test,predd);
accu=(np.trace(asd)/np.size(labels_test))*100;
print("accuracy layer 6......")
print(accu)
#print(classification_report(labels_test,predd,target_names=dirs))
#%% maximum a posterioir estimate

#%% weighted probabilites
'''
prob4_pred=weight[0]*prob4_pred
prob6_pred=weight[1]*prob6_pred
prob7_pred=weight[2]*prob7_pred
prob8_pred=weight[3]*prob8_pred
prob10_pred=weight[4]*prob10_pred
prob11_pred=weight[5]*prob11_pred
prob13_pred=weight[6]*prob13_pred
prob14_pred=weight[7]*prob14_pred
prob16_pred=weight[8]*prob16_pred
prob17_pred=weight[9]*prob17_pred
prob18_pred=weight[10]*prob18_pred
prob20_pred=weight[11]*prob20_pred
prob21_pred=weight[12]*prob21_pred
prob23_pred=weight[13]*prob23_pred
prob24_pred=weight[14]*prob24_pred
'''
#%%



print("Development test data")
k3=np.add(prob23_pred,prob24_pred)
a=np.add(prob21_pred,prob20_pred)
b=np.add(prob18_pred,prob17_pred)
c=np.add(a,b)  # 21-17
d=np.add(c,prob16_pred) #21-16
e=np.add(prob14_pred,prob13_pred)
f=np.add(d,e) #21-13
g=np.add(prob11_pred,prob10_pred) 
h=np.add(f,g) #21-10
i=np.add(prob7_pred,prob6_pred)

k=np.add(h,i) #21-7
#k2=np.add(j,k) # 21-4
#k32=np.add(j,i) #8-4

final=np.add(d,k3)
q1716=np.add(prob17_pred,prob16_pred)


#%%




#k7=np.add(prob4_pred,i)
k8=prob8_pred#np.add(k7,prob8_pred)
k11=np.add(k8,g)
k14=np.add(k11,e)
k17=np.add(k14,q1716)
k18=np.add(k17,prob18_pred)
k21=np.add(k18,a)
k24=np.add(k21,k3)


predd=[]

#%%

new_prob=prob24_pred+prob23_pred+prob21_pred+prob20_pred+prob18_pred+prob17_pred+prob16_pred+prob14_pred+prob13_pred+prob11_pred#+prob10_pred+prob8_pred+prob7_pred+prob4_pred+prob6_pred

#new_prob=1.5*prob21_pred+1.3*prob20_pred+0.45*prob18_pred+1.7*prob17_pred+prob8_pred+prob4_pred+prob7_pred+prob6_pred+prob16_pred+prob14_pred+prob13_pred+0.9*prob11_pred+prob10_pred+1.9*prob24_pred+1.9*prob23_pred

print('after ensemble')
predd=np.argmax(new_prob,1)


target_names = dirs;#['library', 'metro', 'city','car','forest','residen','beach','train','bus','office','park','cafe','tram','home','grocer']
report=(classification_report(labels_test,predd,target_names=dirs))
asd=confusion_matrix(labels_test,predd);
accu=(np.trace(asd)/np.size(labels_test))*100

print(accu) 
#print(report)

#%% probability 
#prob_supervector=np.concatenate((prob4_pred.flatten(),prob6_pred.flatten(),prob7_pred.flatten(),prob8_pred.flatten(),prob10_pred.flatten(),prob11_pred.flatten(),prob13_pred.flatten(),prob14_pred.flatten(),prob16_pred.flatten(),prob17_pred.flatten(),prob18_pred.flatten(),prob20_pred.flatten(),prob21_pred.flatten(),prob23_pred.flatten(),prob24_pred.flatten()),axis=0)
#new_prob=np.reshape(prob_supervector,(289,225))
##%%
#
#probnew_pred=[]
#X6=np.array(normalize(new_prob, norm='l2', axis=1, copy=True, return_norm=False))
##y=np.array(labels)
#clf6 = svm.SVC(kernel='poly',degree=5,gamma=8,decision_function_shape = "ovo",probability=True,random_state=0)
##clf.decision_function_shape = "ovr"
#clf6.fit(X6, y)
#


#%% majority voting
'''
#comb=[sum24_pred,sum23_pred,sum21_pred,sum17_pred,sum18_pred,sum20_pred,sum16_pred,sum14_pred,sum13_pred,sum11_pred,sum10_pred,sum8_pred,sum7_pred,sum6_pred,sum4_pred]
comb=[sum7_pred,sum6_pred,sum4_pred]

# without batch normalization..
#comb=[sum21_pred,sum17_pred,sum18_pred,sum14_pred,sum11_pred,sum8_pred,sum7_pred,sum4_pred]

#comb=[sum4_pred,sum6_pred,sum7_pred,sum8_pred,sum10_pred]
predd=[]
predd=mode(comb)[0][0]




#







target_names = ['library', 'metro', 'city','car','forest','residen','beach','train','bus','office','park','cafe','tram','home','grocer']
report=(classification_report(labels_test,predd,target_names=target_names))
asd=confusion_matrix(labels_test,predd);
accu=(np.trace(asd)/289)*100;
print(accu)
print(report)

#%% Chaalenge dataset testing

print("challenge data........................")
#sum24_pred_challenge=clf24.predict(normalize(np.array( np.array(ex_test_challenge_24)), norm='l2', axis=1, copy=True, return_norm=False))
prob24_pred_c=clf24.predict_proba(normalize( np.array(ex_test_challenge_24), norm='l2', axis=1, copy=True, return_norm=False))

#sum23_pred_challenge=clf23.predict(normalize(np.array(ex_test_challenge_23), norm='l2', axis=1, copy=True, return_norm=False))
prob23_pred_c=clf23.predict_proba(normalize( np.array(ex_test_challenge_23), norm='l2', axis=1, copy=True, return_norm=False))

#sum21_pred_challenge=clf21.predict(normalize(np.array(ex_test_challenge_21), norm='l2', axis=1, copy=True, return_norm=False))
prob21_pred_c=clf21.predict_proba(normalize( np.array(ex_test_challenge_21), norm='l2', axis=1, copy=True, return_norm=False))

#sum20_pred_challenge=clf20.predict(normalize(np.array(ex_test_challenge_20), norm='l2', axis=1, copy=True, return_norm=False))
prob20_pred_c=clf20.predict_proba(normalize( np.array(ex_test_challenge_20), norm='l2', axis=1, copy=True, return_norm=False))


#sum17_pred_challenge=clf17.predict(normalize(np.array(ex_test_challenge_17), norm='l2', axis=1, copy=True, return_norm=False))
prob17_pred_c=clf17.predict_proba(normalize( np.array(ex_test_challenge_17), norm='l2', axis=1, copy=True, return_norm=False))



#sum16_pred_challenge=clf16.predict(normalize(np.array(ex_test_challenge_16), norm='l2', axis=1, copy=True, return_norm=False))
prob16_pred_c=clf16.predict_proba(normalize( np.array(ex_test_challenge_16), norm='l2', axis=1, copy=True, return_norm=False))




#sum14_pred_challenge=clf14.predict(normalize(np.array(ex_test_challenge_14), norm='l2', axis=1, copy=True, return_norm=False))
prob14_pred_c=clf14.predict_proba(normalize( np.array(ex_test_challenge_14), norm='l2', axis=1, copy=True, return_norm=False))



#sum13_pred_challenge=clf13.predict(normalize(np.array(ex_test_challenge_13), norm='l2', axis=1, copy=True, return_norm=False))
prob13_pred_c=clf13.predict_proba(normalize( np.array(ex_test_challenge_13), norm='l2', axis=1, copy=True, return_norm=False))


#sum11_pred_challenge=clf11.predict(normalize(np.array(ex_test_challenge_11), norm='l2', axis=1, copy=True, return_norm=False))
prob11_pred_c=clf11.predict_proba(normalize( np.array(ex_test_challenge_11), norm='l2', axis=1, copy=True, return_norm=False))



#sum10_pred_challenge=clf10.predict(normalize(np.array(ex_test_challenge_10), norm='l2', axis=1, copy=True, return_norm=False))
prob10_pred_c=clf10.predict_proba(normalize( np.array(ex_test_challenge_10), norm='l2', axis=1, copy=True, return_norm=False))



#sum18_pred_challenge=clf18.predict(normalize(np.array(ex_test_challenge_18), norm='l2', axis=1, copy=True, return_norm=False))
prob18_pred_c=clf18.predict_proba(normalize( np.array(ex_test_challenge_18), norm='l2', axis=1, copy=True, return_norm=False))

#sum8_pred_challenge=clf8.predict(normalize(np.array(ex_test_challenge_8), norm='l2', axis=1, copy=True, return_norm=False))
prob8_pred_c=clf8.predict_proba(normalize( np.array(ex_test_challenge_8), norm='l2', axis=1, copy=True, return_norm=False))

#sum7_pred_challenge=clf7.predict(normalize(np.array(ex_test_challenge_7), norm='l2', axis=1, copy=True, return_norm=False))
prob7_pred_c=clf7.predict_proba(normalize( np.array(ex_test_challenge_7), norm='l2', axis=1, copy=True, return_norm=False))


#sum6_pred_challenge=clf6.predict(normalize(np.array(ex_test_challenge_6), norm='l2', axis=1, copy=True, return_norm=False))
prob6_pred_c=clf6.predict_proba(normalize( np.array(ex_test_challenge_6), norm='l2', axis=1, copy=True, return_norm=False))



#sum4_pred_challenge=clf4.predict(normalize(np.array(ex_test_challenge_4), norm='l2', axis=1, copy=True, return_norm=False))
prob4_pred_c=clf4.predict_proba(normalize( np.array(ex_test_challenge_4), norm='l2', axis=1, copy=True, return_norm=False))

#%% challenge maximum a posteriori
a_c=np.add(prob21_pred_c,prob20_pred_c)
b_c=np.add(prob18_pred_c,prob17_pred_c)
c_c=np.add(a_c,b_c)
d_c=np.add(c_c,prob16_pred_c)
e_c=np.add(prob14_pred_c,prob13_pred_c)
f_c=np.add(d_c,e_c)
g_c=np.add(prob11_pred_c,prob10_pred_c)
h_c=np.add(f_c,g_c)
i_c=np.add(prob8_pred_c,prob7_pred_c)
j_c=np.add(prob6_pred_c,prob4_pred_c)
k_c=np.add(h_c,i_c)
k2_c=np.add(j_c,k_c)

k3_c=np.add(prob23_pred_c,prob24_pred_c)
k4_c=np.add(k3_c,k2_c)
predd=[]
predd=np.argmax(k4_c,1)
target_names = dirs#['library', 'metro', 'city','car','forest','residen','beach','train','bus','office','park','cafe','tram','home','grocer']
report=(classification_report(labels_test_chal,predd,target_names=target_names))
asd=confusion_matrix(labels_test_chal,predd);
accu=(np.trace(asd)/390)*100;
print(accu)
print(report)






#%% challnege majority

comb=[sum24_pred_challenge,sum23_pred_challenge,sum21_pred_challenge,sum20_pred_challenge,sum17_pred_challenge,sum18_pred_challenge,sum16_pred_challenge,sum14_pred_challenge,sum13_pred_challenge,sum8_pred_challenge,sum7_pred_challenge,sum6_pred_challenge,sum4_pred_challenge]

#without normaliztion
#comb=[sum21_pred_challenge,sum17_pred_challenge,sum18_pred_challenge,sum14_pred_challenge,sum8_pred_challenge,sum7_pred_challenge,sum4_pred_challenge]

predd=[]
predd=mode(comb)[0][0]
target_names = ['library', 'metro', 'city','car','forest','residen','beach','train','bus','office','park','cafe','tram','home','grocer']
report_challenge=(classification_report(labels_test_chal,predd,target_names=target_names))
asd=confusion_matrix(labels_test_chal,predd);
accu=(np.trace(asd)/390)*100;
print(accu)


##%% TEST WITH ENERGY FEASTURES
#
#X4=np.array(normalize(ex4, norm='l2', axis=1, copy=True, return_norm=False))
#y=np.array(labels)
#clf4 = svm.SVC(kernel='poly',degree=5,gamma=15,decision_function_shape = "ovo")
##clf.decision_function_shape = "ovr"
#clf4.fit(X4, y)
#Development test data
after ensemble
78.75
#sum4_pred=clf4.predict(normalize(ex_test4, norm='l2', axis=1, copy=True, return_norm=False))
#
#print(classification_report(labels_test,sum4_pred,target_names=target_names))
#
#
#asd=confusion_matrix(labels_test,sum4_pred);
#accu=(np.trace(asd)/298)*100;
#print(accu)


#%% feature concatenation
train_cat= np.hstack((X4,X6,X7,X8,X10,X11,X13,X14,X16,X17,X18,X20,X21,X23,X24))
test_cat=np.hstack((normalize(ex_test4, norm='l2', axis=1, copy=True, return_norm=False),normalize(ex_test6, norm='l2', axis=1, copy=True, return_norm=False),normalize(ex_test7, norm='l2', axis=1, copy=True, return_norm=False),normalize(ex_test8, norm='l2', axis=1, copy=True, return_norm=False),normalize(ex_test10, norm='l2', axis=1, copy=True, return_norm=False),normalize(ex_test11, norm='l2', axis=1, copy=True, return_norm=False),normalize(ex_test13, norm='l2', axis=1, copy=True, return_norm=False),normalize(ex_test14, norm='l2', axis=1, copy=True, return_norm=False),normalize(ex_test16, norm='l2', axis=1, copy=True, return_norm=False),normalize(ex_test17, norm='l2', axis=1, copy=True, return_norm=False),normalize(ex_test18, norm='l2', axis=1, copy=True, return_norm=False),normalize(ex_test20, norm='l2', axis=1, copy=True, return_norm=False),normalize(ex_test21, norm='l2', axis=1, copy=True, return_norm=False),normalize(ex_test23, norm='l2', axis=1, copy=True, return_norm=False),normalize(ex_test24, norm='l2', axis=1, copy=True, return_norm=False)))

#%%
X5=[]
clf5=[]
prob5_pred=[]
X5=np.array(normalize(X_train, norm='l2', axis=1, copy=True, return_norm=False))
#y=np.array(labels)
clf5 = svm.SVC(kernel='poly',degree=5,gamma=5,decision_function_shape = "ovo",probability=True,random_state=0)
#clf.decision_function_shape = "ovr"
clf5.fit(X5, y)

#sum4_pred=clf4.predict(normalize(ex_test4, norm='l2', axis=1, copy=True, return_norm=False))
prob5_pred=clf5.predict_proba(normalize(X_test, norm='l2', axis=1, copy=True, return_norm=False))
predd=[]

predd=np.argmax(prob5_pred,1)


asd=confusion_matrix(labels_test,predd);
accu=(np.trace(asd)/500)*100;
print(accu)
#%% random forest
clf = RandomForestClassifier(n_estimators=100,random_state=0)
clf.fit(train_cat, labels)

pred_prob_test=clf.predict(test_cat)

asd=confusion_matrix(labels_test,pred_prob_test);
accu=(np.trace(asd)/np.size(labels_test))*100;
print(accu)
#%% LLE

import sklearn.manifold as MANIFOLD
clf=MANIFOLD.LocallyLinearEmbedding(n_neighbors=120, n_components=50, reg=0.501, eigen_solver='auto', tol=1e-06, max_iter=100, method='standard', hessian_tol=0.0001, modified_tol=1e-12, neighbors_algorithm='auto', random_state=None, n_jobs=1)
clf.fit(train_cat)
X_train=clf.transform(train_cat)
X_test=clf.transform(test_cat)
'''