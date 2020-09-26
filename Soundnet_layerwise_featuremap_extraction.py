# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 20:53:01 2018

@author: user
"""

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
#import theano 
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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras.layers import LSTM
import scipy.io
import pickle
import os
from keras.models import Model
from keras.models import load_model
#%%
#import numpy as np
param_G=np.load("/home/arshdeep/SoundNet-tensorflow-master/models/sound8.npy", encoding = 'latin1').item()
initia_weights1=[np.reshape(param_G['conv1']['weights'],(64,1,16)),param_G['conv1']['biases'],param_G['conv1']['gamma'],param_G['conv1']['beta'],param_G['conv1']['mean'],param_G['conv1']['var'],np.reshape(param_G['conv2']['weights'],(32,16,32)),param_G['conv2']['biases'],param_G['conv2']['gamma'],param_G['conv2']['beta'],param_G['conv2']['mean'],param_G['conv2']['var'],np.reshape(param_G['conv3']['weights'],(16,32,64)),param_G['conv3']['biases'],param_G['conv3']['gamma'],param_G['conv3']['beta'],param_G['conv3']['mean'],param_G['conv3']['var'],np.reshape(param_G['conv4']['weights'],(8,64,128)),param_G['conv4']['biases'],param_G['conv4']['gamma'],param_G['conv4']['beta'],param_G['conv4']['mean'],param_G['conv4']['var'],np.reshape(param_G['conv5']['weights'],(4,128,256)),param_G['conv5']['biases'],param_G['conv5']['gamma'],param_G['conv5']['beta'],param_G['conv5']['mean'],param_G['conv5']['var'],np.reshape(param_G['conv6']['weights'],(4,256,512)),param_G['conv6']['biases'],param_G['conv6']['gamma'],param_G['conv6']['beta'],param_G['conv6']['mean'],param_G['conv6']['var'],np.reshape(param_G['conv7']['weights'],(4,512,1024)),param_G['conv7']['biases'],param_G['conv7']['gamma'],param_G['conv7']['beta'],param_G['conv7']['mean'],param_G['conv7']['var']]#,np.reshape(param_G['conv8']['weights'],(8,1024,1000)),param_G['conv8']['biases'],np.reshape(param_G['conv8_2']['weights'],(8,1024,401)),param_G['conv8_2']['biases']]



#%%

model =Sequential()

model.add(Conv1D(16,64,strides=2,input_shape=(44100*6,1))) #layer1
model.add(ZeroPadding1D(padding=16))
model.add(BatchNormalization()) #layer2
convout1= Activation('relu')
model.add(convout1) #layer3


#initia_weights1=[np.reshape(param_G['conv1']['weights'],(64,1,16)),param_G['conv1']['biases'],param_G['conv1']['gamma'],param_G['conv1']['beta'],param_G['conv1']['mean'],param_G['conv1']['var'],np.reshape(param_G['conv2']['weights'],(32,16,32)),param_G['conv2']['biases'],param_G['conv2']['gamma'],param_G['conv2']['beta'],param_G['conv2']['mean'],param_G['conv2']['var'],np.reshape(param_G['conv3']['weights'],(16,32,64)),param_G['conv3']['biases'],param_G['conv3']['gamma'],param_G['conv3']['beta'],param_G['conv3']['mean'],param_G['conv3']['var'],np.reshape(param_G['conv4']['weights'],(8,64,128)),param_G['conv4']['biases'],param_G['conv4']['gamma'],param_G['conv4']['beta'],param_G['conv4']['mean'],param_G['conv4']['var'],np.reshape(param_G['conv5']['weights'],(4,128,256)),param_G['conv5']['biases'],param_G['conv5']['gamma'],param_G['conv5']['beta'],param_G['conv5']['mean'],param_G['conv5']['var'],np.reshape(param_G['conv6']['weights'],(4,256,512)),param_G['conv6']['biases'],param_G['conv6']['gamma'],param_G['conv6']['beta'],param_G['conv6']['mean'],param_G['conv6']['var'],np.reshape(param_G['conv7']['weights'],(4,512,1024)),param_G['conv7']['biases'],param_G['conv7']['gamma'],param_G['conv7']['beta'],param_G['conv7']['mean'],param_G['conv7']['var'],np.reshape(param_G['conv8']['weights'],(8,1024,1000)),param_G['conv8']['biases'],np.reshape(param_G['conv8_2']['weights'],(8,1024,401)),param_G['conv8_2']['biases']]

model.add(MaxPooling1D(pool_size=8, padding='valid')) #layer4
#
#
model.add(Conv1D(32,32,strides=2)) #layer5
model.add(ZeroPadding1D(padding=8))
model.add(BatchNormalization()) #layer6
convout2= Activation('relu')
model.add(convout2) #layer7
#model.add(Dropout(0.5))

model.add(MaxPooling1D(pool_size=8, padding='valid')) #layer8

model.add(Conv1D(64,16,strides=2)) #layer9
model.add(ZeroPadding1D(padding=4))
model.add(BatchNormalization()) #layer10
convout3= Activation('relu')
model.add(convout3) #layer11
#model.add(Dropout(0.5))

model.add(Conv1D(128,8,strides=2)) #layer12
model.add(ZeroPadding1D(padding=2))
model.add(BatchNormalization()) #layer13
convout4= Activation('relu')
model.add(convout4) #layer14


model.add(Conv1D(256,4,strides=2)) #layer15
model.add(ZeroPadding1D(padding=1))
model.add(BatchNormalization()) #layer16
convout5= Activation('relu')
model.add(convout5) #layer17

model.add(MaxPooling1D(pool_size=4,padding='valid')) #layer18

model.add(Conv1D(512,4,strides=2)) #layer15
model.add(ZeroPadding1D(padding=1))
model.add(BatchNormalization()) #layer16
convout6= Activation('relu')
model.add(convout6) #layer17
#model.add(Dropout(0.5))
#model.set_weights(initia_weights1)
model.add(Conv1D(1024,4,strides=2)) #layer18
model.add(ZeroPadding1D(padding=1))
model.add(BatchNormalization()) #layer19
convout7= Activation('relu')
model.add(convout7) #layer2/home/surbhi/surbhi/DATA_BABY0
#model.add(Dropout(0.5))

model.set_weights(initia_weights1)
#model.add(Conv1D(1000,8,strides=2,padding='valid')) #layer21
#
##model.add(BatchNormalization()) #layer22
#convout8= Activation('relu')
#model.add(convout8) #layer23
#
#model.add(Conv1D(401,8,strides=2,padding='valid')) #layer21
#
##model.add(BatchNormalization()) #layer22
#convout8_2= Activation('relu')
#model.add(convout8_2) #layer23
#%%
#model.set_weights(initia_weights1)
model.add(Flatten())
model.add((Dense(100)))
model.add((Activation('relu')))
model.add(Dropout(0.1))


model.add(Dense(15))
model.add(Activation('softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

#model.load_weights('/home/arshdeep/DCASE2018_FINETUNE/best_model_dcase18_SoundNet.h5py')

#%%
import librosa

#%% load_model
#base_model=model

#%%
#layer_4=[];layer_6=[];layer_7=[];layer_8=[];layer_10=[];layer_11=[];layer_13=[];layer_14=[];layer_16=[];layer_17=[];layer_18=[];layer_20=[];layer_21=[];layer_23=[];layer_24=[];

#layers=['max_pooling1d_1','batch_normalization_2','activation_2','max_pooling1d_2','batch_normalization_3','activation_3','batch_normalization_4','activation_4','batch_normalization_5','activation_5','max_pooling1d_3','batch_normalization_6','activation_6','batch_normalization_7','activation_7']
#layers=['activation_7','batch_normalization_7']
#name=[layer_4,layer_6,layer_7,layer_8,layer_10,layer_11,layer_13,layer_14,layer_16,layer_17,layer_18,layer_20,layer_21,layer_23,layer_24]
#feature_conv1d7=model1.predict(x_train)

#layers=['activation_7']
#name1=[4,6,7,8,10,11,13,14,16,17,18,20,21,23,24]
#name1=[24,23]
#%%

def feature_save(layer_name,example,part,base_model):
		model_sub_layer=base_model
		feature=model_sub_layer.predict(example)
		filename=str(layer_name)+'_'+part[:-4]
		path=os.path.join('DCASE18_finetune_feature',filename)#+str([:-4])
		np.save(filename,np.reshape(feature,[np.shape(feature)[1],np.shape(feature)[2]]))
		return 
	

#%%
		
def load_audio(audio_path, sr=None):
    # By default, librosa will resample the signal to 22050Hz(sr=None). And range in (-1., 1.)
    sound_sample, sr = librosa.load(audio_path, sr=44100, mono=False,duration=6.0)

    return sound_sample, sr

#%% extract from different layers and save feature
base_model_17=Model(inputs=model.input, outputs=model.get_layer('activation_5').output)
base_model_16=Model(inputs=model.input, outputs=model.get_layer('batch_normalization_5').output)
base_model_14=Model(inputs=model.input, outputs=model.get_layer('activation_4').output)
base_model_13=Model(inputs=model.input, outputs=model.get_layer('batch_normalization_4').output)
base_model_11=Model(inputs=model.input, outputs=model.get_layer('activation_3').output)
base_model_10=Model(inputs=model.input, outputs=model.get_layer('batch_normalization_3').output)
base_model_4=Model(inputs=model.input, outputs=model.get_layer('max_pooling1d_1').output)
base_model_6=Model(inputs=model.input, outputs=model.get_layer('batch_normalization_2').output)
base_model_7=Model(inputs=model.input, outputs=model.get_layer('activation_2').output)
base_model_8=Model(inputs=model.input, outputs=model.get_layer('max_pooling1d_2').output)
base_model_21=Model(inputs=model.input, outputs=model.get_layer('activation_6').output)
base_model_20=Model(inputs=model.input, outputs=model.get_layer('batch_normalization_6').output)
base_model_18=Model(inputs=model.input, outputs=model.get_layer('max_pooling1d_3').output)
base_model_24=Model(inputs=model.input, outputs=model.get_layer('activation_7').output)
base_model_23=Model(inputs=model.input, outputs=model.get_layer('batch_normalization_7').output)


#base_model_3=Model(inputs=model.input, outputs=model.get_layer('conv1d_1').output)


#%%			

os.chdir('/media/arshdeep/B294A78494A749A52/DCASE_2019_SOUNDNetfeatures/dcase2019_leader_board_features')

for root, dirs, files in os.walk("/media/arshdeep/B294A78494A749A52/DCASE_2019_SOUNDNetfeatures/TAU-urban-acoustic-scenes-2019-openset-leaderboard", topdown=False):
	for name in dirs:
		parts = []
		parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith('.wav')]
		print(name, "...")
		k=0
		for part in parts:
			k=k+1
			print(k,part)
			file_name3='3_'+part[:-4]+'.npy'
#			file_name10='10_'+part[:-4]+'.npy'
#			file_name6='6_'+part[:-4]+'.npy'
#			file_name14='14_'+part[:-4]+'.npy'
#			file_name13='13_'+part[:-4]+'.npy'
#			file_name10='10_'+part[:-4]+'.npy'
			if (os.path.isfile(file_name3)):# and os.path.isfile(file_name7) and os.path.isfile(file_name6)): #and os.path.isfile(file_name13) and os.path.isfile(file_name11) and os.path.isfile(file_name10)):
				print('file alredy exists')
			else:
					sound_sample,sr =load_audio(os.path.join(root,name,part))#scipy.io.loadmat('data1000_6k_15k.mat')['x']
					sound_sample *= 256
					example=np.reshape(np.array(sound_sample),[1,np.size(sound_sample),1])
					feature24=base_model_24.predict(example)
					feature23=base_model_23.predict(example)
					feature21=base_model_21.predict(example)
					feature20=base_model_20.predict(example)
					feature18=base_model_18.predict(example)
#					feature17=base_model_17.predict(example)
#					feature16=base_model_16.predict(example)
#					feature14=base_model_14.predict(example)
#					feature13=base_model_13.predict(example)
#					feature11=base_model_11.predict(example)
#					feature10=base_model_10.predict(example)
#					feature8=base_model_8.predict(example)
#					feature7=base_model_7.predict(example)
#					feature6=base_model_6.predict(example)
#					feature4=base_model_4.predict(example)	
#					feature3=base_model_3.predict(example)	

				
					filename24='24'+'_'+part[:-4]
					filename23='23'+'_'+part[:-4]
					filename21='21'+'_'+part[:-4]
					filename20='20'+'_'+part[:-4]
					filename18='18'+'_'+part[:-4]
					filename17='17'+'_'+part[:-4]
					filename16='16'+'_'+part[:-4]
					filename14='14'+'_'+part[:-4]
					filename13='13'+'_'+part[:-4]
					filename11='11'+'_'+part[:-4]
					filename10='10'+'_'+part[:-4]
					filename8='8'+'_'+part[:-4]
					filename6='6'+'_'+part[:-4]
					filename7='7'+'_'+part[:-4]
					filename4='4'+'_'+part[:-4]					
					filename21='21'+'_'+part[:-4]
#					filename3='3'+'_'+part[:-4]							
#					path23=os.path.join('DCASE18_finetune_feature',filename23)#+str([:-4])
#					path21=os.path.join('DCASE18_finetune_feature',filename21)#+str([:-4])
#					path20=os.path.join('DCASE18_finetune_feature',filename20)#+str([:-4])
				
					np.save(filename24,np.reshape(feature24,[np.shape(feature24)[1],np.shape(feature24)[2]]))
					np.save(filename23,np.reshape(feature23,[np.shape(feature23)[1],np.shape(feature23)[2]]))
					np.save(filename21,np.reshape(feature21,[np.shape(feature21)[1],np.shape(feature21)[2]]))	
					np.save(filename20,np.reshape(feature20,[np.shape(feature20)[1],np.shape(feature20)[2]]))					
					np.save(filename18,np.reshape(feature18,[np.shape(feature18)[1],np.shape(feature18)[2]]))
#					np.save(filename17,np.reshape(feature17,[np.shape(feature17)[1],np.shape(feature17)[2]]))
#					np.save(filename16,np.reshape(feature16,[np.shape(feature16)[1],np.shape(feature16)[2]]))
###
					np.save(filename11,np.reshape(feature11,[np.shape(feature11)[1],np.shape(feature11)[2]]))
					np.save(filename4,np.reshape(feature4,[np.shape(feature4)[1],np.shape(feature4)[2]]))
					np.save(filename7,np.reshape(feature7,[np.shape(feature7)[1],np.shape(feature7)[2]]))
					np.save(filename6,np.reshape(feature6,[np.shape(feature6)[1],np.shape(feature6)[2]]))
					np.save(filename8,np.reshape(feature8,[np.shape(feature8)[1],np.shape(feature8)[2]]))
					np.save(filename14,np.reshape(feature14,[np.shape(feature14)[1],np.shape(feature14)[2]]))
					np.save(filename13,np.reshape(feature13,[np.shape(feature13)[1],np.shape(feature13)[2]]))
					np.save(filename10,np.reshape(feature10,[np.shape(feature10)[1],np.shape(feature10)[2]]))
#					np.save(filename3,np.reshape(feature3,[np.shape(feature3)[1],np.shape(feature3)[2]]))


			#feature_save(layer_name,example,part,base_model)



#%%
'''	
import scipy.io
scipy.io.savemat(filename24, mdict={'arr': np.reshape(feature24,[np.shape(feature24)[1],np.shape(feature24)[2]])})
scipy.io.savemat(filename4, mdict={'arr': np.reshape(feature4,[np.shape(feature4)[1],np.shape(feature4)[2]])})
scipy.io.savemat(filename7, mdict={'arr': np.reshape(feature7,[np.shape(feature7)[1],np.shape(feature7)[2]])})
scipy.io.savemat(filename11, mdict={'arr': np.reshape(feature11,[np.shape(feature11)[1],np.shape(feature11)[2]])})
scipy.io.savemat(filename14, mdict={'arr': np.reshape(feature14,[np.shape(feature14)[1],np.shape(feature14)[2]])})
scipy.io.savemat(filename17, mdict={'arr': np.reshape(feature17,[np.shape(feature17)[1],np.shape(feature17)[2]])})
scipy.io.savemat(filename21, mdict={'arr': np.reshape(feature21,[np.shape(feature21)[1],np.shape(feature21)[2]])})
scipy.io.savemat(filename3, mdict={'arr': np.reshape(feature3,[np.shape(feature3)[1],np.shape(feature3)[2]])})
'''