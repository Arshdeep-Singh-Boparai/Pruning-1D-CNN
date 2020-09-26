#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 14:28:24 2018

@author: arshdeep
"""

import matplotlib.pyplot as plt
import scipy.io
import os
from scipy.misc import imread
import numpy as np
#from scipy.linalg import hadamard, subspace_angles

cl = 0.
sum = 0
q = 0
j=0
p=0
h=0
#ex4=[]
ex4=[]
ex6=[]
ex7=[]
ex8=[];ex10=[];
ex11=[];ex13=[];ex14=[];ex16=[];ex17=[];
ex18=[];
ex20=[];ex21=[];ex23=[];ex24=[];
#A=[24]#,4,6,7,86,7,8,10,11,13,14]
#A=[10,11,17,21,24]
#A=[24]
A=[4,6,7,8,10,11,13,14,16,17,18,20,21,23,24]

ds=[ex4,ex6,ex7,ex8,ex10,ex11,ex13,ex14,ex16,ex17,ex18,ex20,ex21,ex23,ex24]#ex4,ex6,ex7,ex8,


fold_inf="/home/arshdeep/GM_filter_pruning/Fine_tuned_embeddings/ESC/fold5"

for i in range(np.size(A)):
	j=A[i]
	dict_filename= "pruned_layer"+str(j)+"_dict.npy"#"layer"+str(j)+"_acdl.mat"#"pruned_layer"+str(j)+"_dict.npy"
	layer="layer_"+str(j);
	print(layer)
	labels=[]
	cl=0
	train_path=os.path.join(fold_inf,layer,"train_data")
	for root, dirs, files in os.walk(train_path, topdown=False):
		
		for name in dirs:
			if name in dirs:
				parts = []
				parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith('.npy')]
				print(name, "...")
				for part in parts:
					img = np.load(os.path.join(root,name,part))
					ds[i].append(img)#np.sum(img,0))
					labels.append(cl)
					sum += 1
					j=j+1
				cl +=1



#%%.....................................................................................................................................................		

print('test_data...........................................')
ex_test4=[]
ex_test6=[]
ex_test7=[]
ex_test8=[];ex_test10=[];
ex_test11=[];ex_test13=[];ex_test14=[];ex_test16=[];
ex_test17=[];ex_test18=[];
ex_test20=[];ex_test21=[];ex_test23=[];ex_test24=[];
cl_test=0

ds_test=[ex_test4,ex_test6,ex_test7,ex_test8,ex_test10,ex_test11,ex_test13,ex_test14,ex_test16,ex_test17,ex_test18,ex_test20,ex_test21,ex_test23,ex_test24]
for i in range(np.size(A)):
	j=A[i]
	dict_filename= "pruned_layer"+str(j)+"_dict.npy"#"layer"+str(j)+"_acdl.mat" #
	layer="layer_"+str(j); 
	print(layer)
	labels_test=[]
	cl_test=0		
	test_path=os.path.join(fold_inf,layer,"test_data")
	for root, dirs, files in os.walk(test_path, topdown=False):
		
		for name in dirs:
			if name in dirs:
				parts = []
				parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith('.npy')]
				print(name, "...")
				for part in parts:
					img = np.load(os.path.join(root,name,part))
					ds_test[i].append(img)#np.sum(img,0))#example.flatten())#np.sum(example,0))#np.sum(example,0))#example.flatten())#np.sum(example,0))#,0))#np.sum(img,0))#img.flatten())#np.reshape(img,[img.shape[1],img.shape[0],1]).astype('float'))
					feature=[]
					labels_test.append(cl_test)
					sum += 1
					j=j+1
				cl_test +=1

