


import numpy as np
import librosa
import os
import scipy.io
from scipy import signal
from scipy.misc import imread
from scipy.stats import mode
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sklearn.manifold as MANIFOLD
#%% conacatenate features from all layers 
con_training_data=[]
con_testing_data=[]
#A=[4,6,7,8,10,11,13,14,16,17,18,20,21,23,24]
A=[4,7,8,11,14,17,18,21,24]
for i in range(9):
    i=A[i]
    layer="layer"+str(i);
    train_filename="trian_data_layer"+str(i)+".mat"
    test_filename="test_data_layer"+str(i)+".mat"

    train_path=os.path.join("/home/arshdeep/dict_feat_fold2_new",train_filename)
    test_path=os.path.join("/home/arshdeep/dict_feat_fold2_new",test_filename)


    data=scipy.io.loadmat(train_path)['arr']
    data_test=scipy.io.loadmat(test_path)['arr']
    
    con_training_data.append((normalize(data, norm='l2', axis=1, copy=True, return_norm=False)))
    con_testing_data.append((normalize(data_test, norm='l2', axis=1, copy=True, return_norm=False)))
    data=[]
    data_test=[]

#%% build trainig and testing data
    
    
train_data=np.hstack(con_training_data)
test_data=np.hstack(con_testing_data)    
labels_train= scipy.io.loadmat('/home/arshdeep/dict_feat_fold2_new/labels_train.mat')['arr'][0,:]
labels_test= scipy.io.loadmat('/home/arshdeep/dict_feat_fold2_new/labels_test.mat')['arr'][0,:]    


#%% DImenasionality reduction 





#%% SSVM
'''
X23=np.array(train_data)#normalize(train_data, norm='l2', axis=1, copy=True, return_norm=False))
#y=np.array(labels_train)
clf23 = svm.SVC(kernel='poly',degree=8,gamma=3,decision_function_shape = "ovo",probability=True)
#clf.decision_function_shape = "ovr"
clf23.fit(X23, labels_train)
sum23_pred=clf23.predict(test_data)#normalize(test_data, norm='l2', axis=1, copy=True, return_norm=False))
#prob23_pred_svm18=clf23.predict_proba((test_data[:,4336-2048-512-512-256:4336-2048-512-256]))#, norm='max', axis=1, copy=True, return_norm=False))


#target_names = ['library', 'metro', 'city','car','forest','residen','beach','train','bus','office','park','cafe','tram','home','grocer']
print(classification_report(labels_test,sum23_pred,target_names=target_names))
asd=confusion_matrix(labels_test,sum23_pred);
accu=(np.trace(asd)/289)*100;
print(accu)

#%% Random forest

clf = RandomForestClassifier(n_estimators=800,random_state=0)
clf.fit(train_data, labels_train)

pred_prob_test1=clf.predict(test_data)
#pred_prob_train1=clf.predict_proba(train_data)

asd=confusion_matrix(labels_test,pred_prob_test1);

target_names=['train',	'residential_area',	'forest path',	'café',	'office'	,'library'	,'beach','car','tram',	'metro','bus','city'	,'grocery_store','park',	'home'];
print(classification_report(labels_test,pred_prob_test1,target_names=target_names))

accu=(np.trace(asd)/289)*100;
print(accu)

#%%
wrong_example_index=[]
#scipy.io.savemat('prob_train.mat', mdict={'arr': np.array(pred_prob_train)})      
for i in range(289):
    if labels_test[i]!=pred_labels_test[i]:
        wrong_example_index.append(i)
        

#%% t-sne plot
        

import numpy as np
from sklearn.manifold import TSNE
#train_data=pred_prob_train;
train = train_data[0:60,:];
residence= train_data[60:119,:];
forest= train_data[119:176,:]
cafe=train_data[176:236,:]


office = train_data[236:295,:];
lib= train_data[295:352,:];
beach= train_data[352:411,:]
car=train_data[411:469,:]

tram = train_data[469:529,:];
metro= train_data[529:588,:];
bus= train_data[588:647,:]
city=train_data[647:707,:]
grocery = train_data[707:766,:];
park= train_data[766:824,:];
home= train_data[824:,:]


X= np.vstack((train,residence,forest,cafe,office,lib,beach,car,tram,metro,bus,city,grocery,park,home))


labels=np.reshape(labels_train,[880])
label=np.hstack((labels[0:60],labels[60:119],labels[119:176],labels[176:236],labels[236:295],labels[295:352],labels[352:411],labels[411:469],labels[469:529],labels[529:588],labels[588:647],labels[647:707],labels[707:766],labels[766:824],labels[824:]))

X_embedded = TSNE(n_components=2,perplexity=60,n_iter=800).fit_transform(X)
X_embedded.shape
#label=np.reshape(labels_train,880)
#%% tsne for matlab dict data set with train, residential area..
fig=plt.figure(figsize=(18, 15))
ax=plt.subplot(111)

plt.scatter(X_embedded[:,0],X_embedded[:,1],c=y,s=800)


#%%
l1=plt.scatter(X_embedded[0:60,0],X_embedded[0:60,0],marker='*',s=800)
l2=plt.scatter(X_embedded[60:119,0],X_embedded[60:119,1],marker='^',s=800)
l3=plt.scatter(X_embedded[119:176,0],X_embedded[119:176,1],marker='+',s=800)
l4=plt.scatter(X_embedded[176:236:,0],X_embedded[176:236:,1],s=800)

l5=plt.scatter(X_embedded[236:295,0],X_embedded[236:295,0],marker='*',s=800)
l6=plt.scatter(X_embedded[295:352,0],X_embedded[295:352,1],marker='^',s=800)
l7=plt.scatter(X_embedded[352:411,0],X_embedded[352:411,1],marker='+',s=800)
l8=plt.scatter(X_embedded[411:469:,0],X_embedded[411:469:,1],s=800)

l9=plt.scatter(X_embedded[469:529,0],X_embedded[469:529,0],marker='*',s=800)
l10=plt.scatter(X_embedded[529:588,0],X_embedded[529:588,1],marker='^',s=800)
l11=plt.scatter(X_embedded[588:647,0],X_embedded[588:647,1],marker='+',s=800)
l12=plt.scatter(X_embedded[647:707:,0],X_embedded[647:707:,1],s=800)

l13=plt.scatter(X_embedded[707:766,0],X_embedded[707:766,0],marker='*',s=800)
l14=plt.scatter(X_embedded[766:824,0],X_embedded[766:824,1],marker='^',s=800)
l15=plt.scatter(X_embedded[824:,0],X_embedded[824:,1],marker='+',s=800)




#plt.legend((l1,l2,l3),(  'Beach','library','tram'),scatterpoints=1,loc='below right',ncol=3,fontsize=30)
#plt.legend((l1,l2,l3),(  'Beach','library','tram'),loc='center left', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5,fontsize=18)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
#ax.legend((l1,l2,l3),(  'Beach','library','tram'),loc='upper center', bbox_to_anchor=(0.9, 1),fontsize=35)
ax.legend((l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15),(  'train',	'residential_area',	'forest path',	'café',	'office'	,'library'	,'beach','car','tram',	'metro','bus','city'	,'grocery_store','park',	'home'),loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=6, fancybox=True,fontsize=10)

#%% test data t-sne

train_data=test_data;
train = train_data[0:18,:];
residence= train_data[18:37,:];
forest= train_data[37:58,:]
cafe=train_data[58:76,:]


office = train_data[76:95,:];
lib= train_data[95:115,:];
beach= train_data[115:134,:]
car=train_data[134:154,:]

tram = train_data[154:172,:];
metro= train_data[172:191,:];
bus= train_data[191:210,:]
city=train_data[210:228,:]
grocery = train_data[228:247,:];
park= train_data[247:267,:];
home= train_data[267:,:]


X= np.vstack((train,residence,forest,cafe,office,lib,beach,car,tram,metro,bus,city,grocery,park,home))


labels=np.reshape(labels_test,[289])
label=np.hstack((labels[0:18],labels[18:37],labels[37:58],labels[58:76],labels[76:95],labels[95:115],labels[115:134],labels[134:154],labels[154:172],labels[172:191],labels[191:210],labels[210:228],labels[228:247],labels[247:267],labels[267:]))

X_embedded = TSNE(n_components=2,perplexity=24,n_iter=800).fit_transform(X)
X_embedded.shape

fig=plt.figure(figsize=(18, 15))
ax=plt.subplot(111)
l1=plt.scatter(X_embedded[0:18,0],X_embedded[0:18,0],marker='*',s=800)
l2=plt.scatter(X_embedded[18:37,0],X_embedded[18:37,1],marker='^',s=800)
l3=plt.scatter(X_embedded[37:58,0],X_embedded[37:58,1],marker='+',s=800)
l4=plt.scatter(X_embedded[58:76,0],X_embedded[58:76,1],s=800)

l5=plt.scatter(X_embedded[76:95,0],X_embedded[76:95,0],marker='*',s=800)
l6=plt.scatter(X_embedded[95:115,0],X_embedded[95:115,1],marker='^',s=800)
l7=plt.scatter(X_embedded[115:134,0],X_embedded[115:134,1],marker='+',s=800)
l8=plt.scatter(X_embedded[134:154,0],X_embedded[134:154:,1],s=800)

l9=plt.scatter(X_embedded[154:172,0],X_embedded[154:172,0],marker='*',s=800)
l10=plt.scatter(X_embedded[172:191,0],X_embedded[172:191,1],marker='^',s=800)
l11=plt.scatter(X_embedded[191:210,0],X_embedded[191:210,1],marker='+',s=800)
l12=plt.scatter(X_embedded[210:228,0],X_embedded[210:228,1],s=800)

l13=plt.scatter(X_embedded[228:247,0],X_embedded[228:247,0],marker='*',s=800)
l14=plt.scatter(X_embedded[247:267,0],X_embedded[247:267,1],marker='^',s=800)
l15=plt.scatter(X_embedded[267:,0],X_embedded[267:,1],marker='+',s=800)




#plt.legend((l1,l2,l3),(  'Beach','library','tram'),scatterpoints=1,loc='below right',ncol=3,fontsize=30)
#plt.legend((l1,l2,l3),(  'Beach','library','tram'),loc='center left', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5,fontsize=18)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
#ax.legend((l1,l2,l3),(  'Beach','library','tram'),loc='upper center', bbox_to_anchor=(0.9, 1),fontsize=35)
ax.legend((l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15),(  'train',	'residential_area',	'forest path',	'café',	'office'	,'library'	,'beach','car','tram',	'metro','bus','city'	,'grocery_store','park',	'home'),loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=6, fancybox=True,fontsize=10)


#%% PCA

import numpy as np
from sklearn.decomposition import PCA
X = train_data;
pca = PCA(n_components=150)
pca.fit(X)

pca_train_data=pca.transform(X)
pca_test_data=pca.transform(test_data)


#%%
clf = RandomForestClassifier(n_estimators=100,random_state=0)
clf.fit(pca_train_data, labels_train)

pred_prob_test=clf.predict(pca_test_data)


asd=confusion_matrix(labels_test,pred_prob_test);

target_names=['train',	'residential_area',	'forest path',	'café',	'office'	,'library'	,'beach','car','tram',	'metro','bus','city'	,'grocery_store','park',	'home'];
print(classification_report(labels_test,pred_prob_test,target_names=target_names))

accu=(np.trace(asd)/289)*100;
print(accu)



#%%

X=np.array(prob_train)
X_embedded = TSNE(n_components=2,perplexity=60,n_iter=800).fit_transform(X)
X_embedded.shape
fig=plt.figure(figsize=(18, 15))
ax=plt.subplot(111)

l1=plt.scatter(X_embedded[0:57,0],X_embedded[0:57,0],marker='*',s=800)
l2=plt.scatter(X_embedded[57:116,0],X_embedded[57:116,1],marker='^',s=800)
l3=plt.scatter(X_embedded[116:176,0],X_embedded[116:176,1],marker='+',s=800)
l4=plt.scatter(X_embedded[176:234,0],X_embedded[176:234:,1],s=800)

l5=plt.scatter(X_embedded[234:291,0],X_embedded[234:291,0],marker='*',s=800)
l6=plt.scatter(X_embedded[291:350,0],X_embedded[291:350,1],marker='^',s=800)
l7=plt.scatter(X_embedded[350:409,0],X_embedded[350:409,1],marker='+',s=800)
l8=plt.scatter(X_embedded[409:469:,0],X_embedded[409:469:,1],s=800)

l9=plt.scatter(X_embedded[469:528,0],X_embedded[469:528,0],marker='*',s=800)
l10=plt.scatter(X_embedded[528:587,0],X_embedded[528:587,1],marker='^',s=800)
l11=plt.scatter(X_embedded[587:645,0],X_embedded[587:645,1],marker='+',s=800)
l12=plt.scatter(X_embedded[645:705,0],X_embedded[645:705,1],s=800)

l13=plt.scatter(X_embedded[705:765,0],X_embedded[705:765,0],marker='*',s=800)
l14=plt.scatter(X_embedded[765:821,0],X_embedded[765:821,1],marker='^',s=800)
l15=plt.scatter(X_embedded[821:,0],X_embedded[821:,1],marker='+',s=800)




#plt.legend((l1,l2,l3),(  'Beach','library','tram'),scatterpoints=1,loc='below right',ncol=3,fontsize=30)
#plt.legend((l1,l2,l3),(  'Beach','library','tram'),loc='center left', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5,fontsize=18)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
#ax.legend((l1,l2,l3),(  'Beach','library','tram'),loc='upper center', bbox_to_anchor=(0.9, 1),fontsize=35)
ax.legend((l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15),dirs,loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=6, fancybox=True,fontsize=10)
'''
#%% Manifold learning
clf=[]
clf=MANIFOLD.LocallyLinearEmbedding(n_neighbors=120, n_components=100, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, method='standard', hessian_tol=0.0001, modified_tol=1e-12, neighbors_algorithm='auto', random_state=None, n_jobs=1)
clf.fit(train_data)
X_train=clf.transform(train_data)
X_test=clf.transform(test_data)
#%%
clf=[]
clf = RandomForestClassifier(n_estimators=800,random_state=0)
clf.fit(normalize(train_data, norm='l2', axis=1, copy=True, return_norm=False), labels_train)

pred_prob_test=clf.predict(normalize(test_data, norm='l2', axis=1, copy=True, return_norm=False))
#pred_prob_test_LLE=clf.predict_proba(test_dat1)

asd=confusion_matrix(labels_test,new_pred_label_trans);

target_names=['train',	'residential_area',	'forest path',	'café',	'office'	,'library'	,'beach','car','tram',	'metro','bus','city'	,'grocery_store','park',	'home'];
print(classification_report(labels_test,new_pred_label_trans,target_names=target_names))

accu=(np.trace(asd)/np.size(labels_test))*100;
print(accu)


#%% probablity save
pred_prob_test=clf.predict_proba(normalize(test_data, norm='l2', axis=1, copy=True, return_norm=False))
pred_prob_train=clf.predict_proba(normalize(train_data, norm='l2', axis=1, copy=True, return_norm=False))

scipy.io.savemat('/home/arshdeep/sound_feat/dict_feat_fold1/prob_fold1_train', mdict={'arr': np.array(pred_prob_train)}) 
scipy.io.savemat('/home/arshdeep/sound_feat/dict_feat_fold1/prob_fold1_test', mdict={'arr': np.array(pred_prob_test)}) 
y_train = keras.utils.to_categorical(labels_train, 15)
scipy.io.savemat('/home/arshdeep/sound_feat/dict_feat_fold1/target_prob', mdict={'arr': np.array(pred_prob_test)}) 

#%% expression

A_mat= np.matmul(pred_prob_train.T,y_train)
B_mat=ds=np.linalg.inv(np.matmul(pred_prob_train.T,pred_prob_train))
A_trans=np.matmul(A_mat, B_mat)

pred_new=np.matmul(pred_prob_test,A_trans.T)
new_pred_label_trans=np.argmax(pred_new,1)
#%%
X23=np.array(normalize(X_train, norm='l2', axis=1, copy=True, return_norm=False))
#y=np.array(labels_train)
clf23 = svm.SVC(kernel='linear',degree=4,gamma=10,decision_function_shape = "ovo",probability=True)
#clf.decision_function_shape = "ovr"
clf23.fit(X23, labels_train)
sum23_pred=clf23.predict(normalize(X_test, norm='l2', axis=1, copy=True, return_norm=False))


#prob_train=clf23.predict_proba(normalize(X_train, norm='l2', axis=1, copy=True, return_norm=False))
#prob_test=clf23.predict_proba(normalize(X_test, norm='l2', axis=1, copy=True, return_norm=False))
#prob23_pred_svm18=clf23.predict_proba((test_data[:,4336-2048-512-512-256:4336-2048-512-256]))#, norm='max', axis=1, copy=True, return_norm=False))


#target_names = ['library', 'metro', 'city','car','forest','residen','beach','train','bus','office','park','cafe','tram','home','grocer']
print(classification_report(labels_test,sum23_pred,target_names=target_names))
asd=confusion_matrix(labels_test,sum23_pred);
accu=(np.trace(asd)/np.size(labels_test))*100;
print(accu)

#%% Spectral embeddings.....
'''
specE= MANIFOLD.SpectralEmbedding(n_components=500, affinity='nearest_neighbors')
X_train=specE.fit_transform(train_data)
X_test=specE.fit_transform(test_data)


#%% ZERO SHOT REGRESSION PROBL
from scipy.spatial import distance
pred_prob_train=train_data
pred_prob_test=test_data
train = np.mean(pred_prob_train[0:60,:],0);
residence= np.mean(pred_prob_train[60:119,:],0);
forest= np.mean(pred_prob_train[119:176,:],0)
cafe=np.mean(pred_prob_train[176:236,:],0)


office = np.mean(pred_prob_train[236:295,:],0);
lib= np.mean(pred_prob_train[295:352,:],0);
beach= np.mean(pred_prob_train[352:411,:],0)
car=np.mean(pred_prob_train[411:469,:],0)

tram = np.mean(pred_prob_train[469:529,:],0);
metro= np.mean(pred_prob_train[529:588,:],0);
bus= np.mean(pred_prob_train[588:647,:],)
city=np.mean(pred_prob_train[647:707,:],0)
grocery = np.mean(pred_prob_train[707:766,:],0);
park= np.mean(pred_prob_train[766:824,:],0);
home= np.mean(pred_prob_train[824:,:],0)

pred_labels=[]
for i in range(289):
    l0=distance.euclidean(train,pred_prob_test[i,:])
    l1=distance.euclidean(residence,pred_prob_test[i,:])
    l2=distance.euclidean(forest,prced_prob_test[i,:])
    l3=distance.euclidean(cafe,pred_prob_test[i,:])
    l4=distance.euclidean(office,pred_prob_test[i,:])
    l5=distance.euclidean(lib,pred_prob_test[i,:])
    l6=distance.euclidean(beach,pred_prob_test[i,:])
    l7=distance.euclidean(car,pred_prob_test[i,:])
    l8=distance.euclidean(tram,pred_prob_test[i,:])
    l9=distance.euclidean(metro,pred_prob_test[i,:])
    l10=distance.euclidean(bus,pred_prob_test[i,:])
    l11=distance.euclidean(city,pred_prob_test[i,:])
    l12=distance.euclidean(grocery,pred_prob_test[i,:])
    l13=distance.euclidean(park,pred_prob_test[i,:])
    l14=distance.euclidean(home,pred_prob_test[i,:])
    pred_labels.append(np.argmin((l0,l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14)))
    


print(classification_report(labels_test,pred_labels,target_names=target_names))
asd=confusion_matrix(labels_test,pred_labels);
accu=(np.trace(asd)/289)*100;
print(accu)
    
#%%
from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np
X = pred_prob_train
y = labels_train
clf = NearestCentroid()
clf.fit(X, y)

pred_labels=(clf.predict(pred_prob_test))


#%% dnn 
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import svm
from scipy.stats import mode
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
import h5py
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D, ZeroPadding1D,AveragePooling1D
from keras import backend as K
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from scipy.stats import mode
import theano 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')
from random import shuffle
from keras.callbacks import ModelCheckpoint
import os
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
#from spp.SpatialPyramidPooling import SpatialPyramidPooling
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

#%%
y_train = keras.utils.to_categorical(labels_train, 15)
y_test=keras.utils.to_categorical(labels_test, 15)
#%%
model =Sequential()

model.add(Dense(200, input_dim=100))
model.add(Activation('relu'))
model.add((Dropout(0.1)))
#model.add(BatchNormalization())
model.add(Dense(240))
model.add(Activation('relu'))
model.add((Dropout(0.5)))
##
#model.add(Dense(512))
#model.add(Activation('relu'))


#
model.add(Dense(15))
model.add(Activation('softmax'))
model.compile(loss=keras.losses.mean_squared_error,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


#%%
model.fit(X_train, y_train, epochs=50, batch_size=32,validation_data=(X_test, y_test),shuffle=True)
score1 = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score1[0])
print('Test accuracy:', score1[1])
#%%
pred_labels=np.argmax(model.predict(X_test),1)
print(classification_report(labels_test,pred_labels,target_names=target_names))
asd=(confusion_matrix(labels_test,pred_labels))
accur=np.trace(asd)/289;
print(accur)
'''