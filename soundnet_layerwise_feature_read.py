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
#A=[16,17,18,20,21,23,24]
#ds=[ex4,ex6,ex7,ex8,ex10,ex11,ex13,ex14]
#ds=[ex8,ex10,ex11,ex13,ex14,ex16,ex17,ex18,ex20,ex21,ex23,ex24]
#ds=[ex16,ex17,ex18,ex20,ex21,ex23,ex24]
#ds=[ex11,ex17,ex21,ex24]
ds=[ex4,ex6,ex7,ex8,ex10,ex11,ex13,ex14,ex16,ex17,ex18,ex20,ex21,ex23,ex24]#ex4,ex6,ex7,ex8,
#ds=[ex24]
#%%
'''
os.chdir('/home/arshdeep/FILTER_SELECTION_WORK_JULY18/rank_based/rank_full_150ex')
 #2.CS 3. DE
sim_index24=np.load('sim_index24.npy')
sim_index23=np.load('sim_index24.npy')
sim_index21=np.load('sim_index21.npy')1
sim_index20=np.load('sim_index21.npy')
sim_index18=np.load('sim_index18.npy')
sim_index16=np.load('sim_index17.npy')
sim_index17=np.load('sim_index17.npy')
sim_index14=np.load('sim_index14.npy')
sim_index13=np.load('sim_index14.npy')


#sim_index8=np.load('sim_index8.npy')

sim_index11=np.load('sim_index11.npy')
sim_index10=np.load('sim_index11.npy')
#j=0
sim_index8=np.arange(0,32)
sim_index7=np.arange(0,32)
sim_index6=np.arange(0,32)
sim_index4=np.arange(0,16)
prun_index=[sim_index4,sim_index6,sim_index7,sim_index8,sim_index10,sim_index11,sim_index13,sim_index14,sim_index16,sim_index17,sim_index18,sim_index20,sim_index21,sim_index23,sim_index24]

'''
#%%
fold_inf="/home/arshdeep/GM_filter_pruning/Fine_tuned_embeddings/ESC/fold5"
#chal_path="/home/arshdeep/DCR_transaction/dcase_fold1/evaluation" 
dict_folder="/home/arshdeep/dictOnfeaturemaps/pruned_dict/dcase_fold4_0.1"
#train_path=os.path.join1"/home/arshdeep/SPL_FINE_TUNE_DATA/esc-50/dcase_layerwise_fold1/fold3",layer,"train_data")
#test_path=os.path.join("/home/arshdeep/SPL_FINE_TUNE_DATA/esc-50/dcase_layerwise_fold1/fold3",layer,"test_data")
for i in range(np.size(A)):
	j=A[i]
	dict_filename= "pruned_layer"+str(j)+"_dict.npy"#"layer"+str(j)+"_acdl.mat"#"pruned_layer"+str(j)+"_dict.npy"
	layer="layer_"+str(j);
	print(layer)
	labels=[]
	cl=0
	dict_path=os.path.join(dict_folder,dict_filename)
#	dict_path2=os.path.join("/home/arshdeep/dictOnfeaturemaps/pruned_dict/esc_fold5_0.1"/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict/home/arshdeep/ACDL/dcase_dictionaries/fold2_dict,dict_filename)	
	train_path=os.path.join(fold_inf,layer,"train_data")
	for root, dirs, files in os.walk(train_path, topdown=False):
		
		for name in dirs:
			if name in dirs:
				parts = []
				parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith('.npy')]
				print(name, "...")
				for part in parts:
					img = np.load(os.path.join(root,name,part))
#					feat=np.matmul(img,np.load(dict_path))#scipy.io.loadmat(dict_path)['D'])#np.load(dict_path))#np.load(dict_path))#np.load(dict_path))#np.load(dict_path))#np.load(dict_path))#scipy.io.loadmat(dict_path)['D'])
	
#				if np.load(dict_path1).size==16:
#					feat=np.matmul(img ,np.hstack((np.reshape(np.load(dict_path1),[16,1]),np.load(dict_path2))))
#				else:
#					feat=np.matmul(img ,np.hstack((np.load(dict_path1),np.load(dict_path2))))
#				feat=np.matmul(img ,np.hstack((np.load(dict_path1),np.load(dict_path2))))				

#				example1=np.sum(img,0)
				
#				example_co1=(np.sum(img,0))
#				example_co21=np.sum(np.sum(img[:,0:np.int(np.size(img,1)/2)],0),0)
#				example_co22=np.sum(np.sum(img[:,np.int(np.size(img,1)/2):],0),0)2
#				example_row1=np.sum(np.sum(img,1),0)
#				example_row21=np.sum(np.sum(img[0:np.int(np.size(img,0)/2),:],1),0)
#				example_row22=np.sum(np.sum(img[np.int(np.size(img,0)/2):,:],1),0) 
##				feature=np.hstack((example_co1,example_co21,example_co22,example_row1,example_row21,example_row22));  
#				example2=np.sum(img[0:np.int(np.size(img,0)/2),:],0)
#				example22=np.sum(img[np.int(np.size(img,0)/2):,:],0);
#				example3=np.sum(img[0:np.int(np.size(img,0)/3),:],0);
#				example32=np.sum(img[np.int(np.size(img,0)/3):np.int(np.size(img,0)/3)+np.int(np.size(img,0)/3),:],0);        
#				example33=np.sum(img[np.int(np.size(img,0)/3)+np.int(np.size(img,0)/3):,:],0);
#				example=np.vstack((example1,example2,example22,example3,example32,example33));	
					ds[i].append(img)#np.sum(img,0))#example.flatten())#np.sum(example,0))#np.sum(example,0))#example.flatten())#np.sum(example,0))#,0))#np.sum(img,0))#img.flatten())#np.reshape(img,[img.shape[1],img.shape[0],1]).astype('float'))
					feature=[]
					labels.append(cl)
					sum += 1
					j=j+1
				cl +=1
#	data_dict=[]
#	for index in range(0,50):
#		sd = [k for k,x in enumerate(labels) if labels[k]== index]
#		data=np.array(ds[i])[sd,:,:]
#		a=np.reshape(data,[np.size(sd)*np.shape(data)[1],np.shape(data)[2]]).T
#		u,s,v=np.linalg.svd(a, full_matrices=False, compute_uv=True)
#		print(index,'rank is  ', np.linalg.matrix_rank(a))
#		data_dict.append(u[:,0:np.linalg.matrix_rank(a)])
#		
#	os.chdir('/home/arshdeep/dictOnfeaturemaps/fold4_dcase_fullrank')
#	data_dict_all=np.hstack(data_dict)
#	filname='layer'+str(A[i])+'_dict'
#	np.save(filname,data_dict_all)
#	print(filname, 'saved')
#	print(np.shape(data_dict_all),'shape')



#%%
'''				
np.save('/home/arshdeep/DCR_transaction/data_tsne_acdl_greedy/acdl_11',np.array(ex11))	
np.save('/home/arshdeep/DCR_transaction/data_tsne_acdl_greedy/acdl_17',np.array(ex17))	
np.save('/home/arshdeep/DCR_transaction/data_tsne_acdl_greedy/acdl_21',np.array(ex21))	
np.save('/home/arshdeep/DCR_transaction/data_tsne_acdl_greedy/acdl_24',np.array(ex24))	
np.save('/home/arshdeep/DCR_transaction/data_tsne_acdl_greedy/dirs',np.array(dirs))	
np.save('/home/arshdeep/DCR_transaction/data_tsne_acdl_greedy/labels',np.array(labels))	
'''
				
#%%%				
'''
data_dict=[]	dict_path2=os.path.join("/home/arshdeep/dictOnfeaturemaps/pruned_dict/esc_fold5_0.1",dict_filename)	
for index in range(0,50):
	sd = [i for i,x in enumerate(labels) if labels[i]== index]
	data=np.array(ds[0])[sd,:,:]
	a=np.reshape(data,[np.size(sd)*np.shape(data)[1],np.shape(data)[2]]).T
	u,s,v=np.linalg.svd(a, full_matrices=False, compute_uv=True)
	print(index,'rank is  ', np.linalg.matrix_rank(a))
	data_dict.append(u[:,0:np.linalg.matrix_rank(a)])
#	if np.linalg.matrix_rank(a)>=50:
#		data_dict.append(u[:,0:50])
#	else:
#		data_dict.append(u[:,0:np.linalg.matrix_rank(a)])
		
	

#%%#%%

print(np.linalg.matrix_rank(a))
#%%		

sd = [i for i,x in enumerate(labels) if labels[i]== 0]
data_1=	np.array(ds[0])[sd,:,:]
a1=np.reshape(data_1,[np.size(sd)*np.shape(ds[0])[1],np.shape(ds[0])[2]]).T
sd1= [i for i,x in enumerate(labels) if labels[i]== 7]
data_2=	np.array(ds[0])[sd1,:,:]
a2=np.reshape(data_2,[np.size(sd1)*np.shape(ds[0])[1],np.shape(ds[0])[2]]).T


plt.plot(np.rad2deg(subspace_angles(a1,a2)),label='17')
plt.legend()
			
			
#%%			

import numpy as np
import scipy.stats as stats
import os
from sklearn.preprocessing import normalize
import math
#%%
layer=str(A[0])
data=np.array(ds[0])
print(layer)
filename='sim_index' +layer

index_sorted_DE=[]
#%%validation dataset to select number of filters
class0=np.reshape(data[0:10,:,:],[10,data[0,:,:].shape[0],data[0,:,:].shape[1]])
class1=np.reshape(data[57:67,:,:],[10,data[0,:,:].shape[0],data[0,:,:].shape[1]])
class2=np.reshape(data[140:150,:,:],[10,data[0,:,:].shape[0],data[0,:,:].shape[1]])
class3=np.reshape(data[187:197,:,:],[10,data[0,:,:].shape[0],data[0,:,:].shape[1]])
class4=np.reshape(data[237:247,:,:],[10,data[0,:,:].shape[0],data[0,:,:].shape[1]])
class5=np.reshape(data[291:301,:,:],[10,data[0,:,:].shape[0],data[0,:,:].shape[1]])
class6=np.reshape(data[394:404,:,:],[10,data[0,:,:].shape[0],data[0,:,:].shape[1]])
class7=np.reshape(data[439:449,:,:],[10,data[0,:,:].shape[0],data[0,:,:].shape[1]])
class8=np.reshape(data[469:479,:,:],[10,data[0,:,:].shape[0],data[0,:,:].shape[1]])
class9=np.reshape(data[542:552,:,:],[10,data[0,:,:].shape[0],data[0,:,:].shape[1]])
class10=np.reshape(data[592:602,:,:],[10,data[0,:,:].shape[0],data[0,:,:].shape[1]])
class11=np.reshape(data[681:691,:,:],[10,data[0,:,:].shape[0],data[0,:,:].shape[1]])
class12=np.reshape(data[701:711,:,:],[10,data[0,:,:].shape[0],data[0,:,:].shape[1]])
class13=np.reshape(data[778:788,:,:],[10,data[0,:,:].shape[0],data[0,:,:].shape[1]])
class14=np.reshape(data[860:870,:,:],[1s=128,0,data[0,:,:].shape[0],data[0,:,:].shape[1]])


validation_data=np.vstack((class0,class1,class2,class3,class4,class5,class6,class7,class8,class9,class10,class11,class12,class13,class14));



data_new=np.reshape(validation_data,(150*np.shape(validation_data)[2],np.shape(validation_data)[1]))
valid_data_normalized=(data_new-np.mean(data_new,0))/np.std(data_new,0)

data_normalized=np.reshape(valid_data_normalized,[150,np.shape(validation_data)[1],np.shape(validation_data)[2]])
#%%
os.chdir('/home/arshdeep/FILTER_SELECTION_WORK_JULY18/rank_based/rank_half_150ex')
rnk_21=[]
index=[]
for i in range(np.shape(data_normalized)[2]):
	filter_i=data_normalized[:,:,i]
	rnk_21.append(np.linalg.matrix_rank(filter_i))
	if rnk_21[i]>=int(np.min(np.shape(filter_i))/2):
		index.append(i)



#%%

plt.figure(figsize=(17,7.8))
sd = [i for i,x in enumerate(rnk_24) if x<15 ]
sd21=[i for i,x in enumerate(rnk_21) if x<15 ]

markers_on=sd[0:16]2
plt.plot(rnk_24[0:20],'o',markevery=markers_on,markersize=20,color='g')
plt.plot(rnk_24[0:20],linewidth=5, label='C7 layer')

markers_on=sd21[0:4]

plt.plot(rnk_21[0:20],'o',markevery=markers_on,markersize=20,color='g')
plt.plot(rnk_21[0:20],linewidth=5,label='C5 layer')


plt.text(1.5,165,'Green points represents redundant feature map indexes',weight='bold', bbox=dict(facecolor='g', alpha=0.5),fontsize='26')
plt.text(0.1,47,'Threshold >= 15',weight='bold', bbox=dict(facecolor='w', alpha=0.5),fontsize='26')
plt.arrow(3,45,0,-30,shape='left',linestyle='--',linewidth=2,color='r')
plt.yticks(np.arange(0,151,15))
plt.xticks(np.arange(0,20,1))
plt.xticks(fontsize=24,weight='bold')
plt.yticks(fontsize=24,weight='bold')
plt.ylim([-10,180])
plt.xlim([-0.5,20])
plt.legend(loc=6,prop={'size': 26})
plt.arrow(-0.5,15,20,0,shape='left',linestyle='--',linewidth=2,color='r')
plt.xlabel('Feature map index',fontsize=40,weight='bold')
plt.ylabel('Rank',fontsize=40,weight='bold')
plt.grid(color='b', linestyle='--', linewidth=0.5)
plt.show()


#%%
plt.plot(index)


ex_test4=[]
ex_test6=[]
ex_test7=[]
ex_test8=[];#ex_test10=[];
ex_test11=[];

sim_index=index
#np.save(filename,np.array(sim_index))		
			
print(np.size(sim_index))
			
			
			

'''		

#%%			
#A=[18]
#ex_test18=[]

print('test_data...........................................')
ex_test4=[]
ex_test6=[]
ex_test7=[]
ex_test8=[];ex_test10=[];
ex_test11=[];ex_test13=[];ex_test14=[];ex_test16=[];
ex_test17=[];ex_test18=[];
ex_test20=[];ex_test21=[];ex_test23=[];ex_test24=[];
cl_test=0
#ex_test17=[]
    		
#ds_test=[ex_test16,ex_test17,ex_test18,ex_test20,ex_test21,ex_test23,ex_test24]
#ds_test=[ex_test20,ex_test21,ex_test23,ex_test24]ex_test4,ex_test6,ex_test7,ex_test8,
#ds_test=[ex_test4,ex_test6,ex_test7,ex_test8,ex_test10,ex_test11,ex_test13,ex_test14,ex_test16,ex_test17,ex_test18,ex_test20,ex_test21,ex_test23,ex_test24]
#ds_test=[ex_test4,ex_test6,ex_test7,ex_test8,ex_test10,ex_test11,ex_test13,ex_test14,ex_test16,ex_test17,ex_test18,ex_test20,ex_test21,ex_test23,ex_test24]
#ds_test=[ex_test4,ex_test6,ex_test7,ex_test8,ex_test24]ex_test4,ex_test6,ex_test7,ex_test8,ex_test4,ex_test6,ex_test7,ex_test8,
ds_test=[ex_test4,ex_test6,ex_test7,ex_test8,ex_test10,ex_test11,ex_test13,ex_test14,ex_test16,ex_test17,ex_test18,ex_test20,ex_test21,ex_test23,ex_test24]
for i in range(np.size(A)):
	j=A[i]
	dict_filename= "pruned_layer"+str(j)+"_dict.npy"#"layer"+str(j)+"_acdl.mat" #
	layer="layer_"+str(j); 
	print(layer)
	labels_test=[]
	cl_test=0
	dict_path=os.path.join(dict_folder,dict_filename)
#	dict_path2=os.path.join("/home/arshdeep/dictOnfeaturemaps/pruned_dict/esc_fold5_0.1",dict_filename)		
	test_path=os.path.join(fold_inf,layer,"test_data")
	for root, dirs, files in os.walk(test_path, topdown=False):
		
		for name in dirs:
			if name in dirs:
				parts = []
				parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith('.npy')]
				print(name, "...")
				for part in parts:
					img = np.load(os.path.join(root,name,part))
#					feat=np.matmul(img,np.load(dict_path))#scipy.io.loadmat(dict_path)['D'])
#	
#				if np.load(dict_path1).size==16:/media/arshdeep/B294A78494A749A52/DCASE_2019_SOUNDNetfeatures/layer_wise_features
#					feat=np.matmul(img ,np.hstack((np.reshape(np.load(dict_path1),[16,1]),np.load(dict_path2))))
#				else:
#					feat=np.matmul(img ,np.hstack((np.load(dict_path1),np.load(dict_path2))))
#				feat=np.matmul(img ,np.hstack((np.load(dict_path1),np.load(dict_path2))))				

#				example1=np.sum(img,0)
				
#				example_co1=(np.sum(img,0))
#				example_co21=np.sum(np.sum(img[:,0:np.int(np.size(img,1)/2)],0),0)
#				example_co22=np.sum(np.sum(img[:,np.int(np.size(img,1)/2):],0),0)2
#				example_row1=np.sum(np.sum(img,1),0)
#				example_row21=np.sum(np.sum(img[0:np.int(np.size(img,0)/2),:],1),0)
#				example_row22=np.sum(np.sum(img[np.int(np.size(img,0)/2):,:],1),0) 
##				feature=np.hstack((example_co1,example_co21,example_co22,example_row1,example_row21,example_row22));  
#				example2=np.sum(img[0:np.int(np.size(img,0)/2),:],0)
#				example22=np.sum(img[np.int(np.size(img,0)/2):,:],0);
#				example3=np.sum(img[0:np.int(np.size(img,0)/3),:],0);
#				example32=np.sum(img[np.int(np.size(img,0)/3):np.int(np.size(img,0)/3)+np.int(np.size(img,0)/3),:],0);        
#				example33=np.sum(img[np.int(np.size(img,0)/3)+np.int(np.size(img,0)/3):,:],0);
#				example=np.vstack((example1,example2,example22,example3,example32,example33));	
					ds_test[i].append(img)#np.sum(img,0))#example.flatten())#np.sum(example,0))#np.sum(example,0))#example.flatten())#np.sum(example,0))#,0))#np.sum(img,0))#img.flatten())#np.reshape(img,[img.shape[1],img.shape[0],1]).astype('float'))
					feature=[]
					labels_test.append(cl_test)
					sum += 1
					j=j+1
				cl_test +=1

#%%		
'''				
os.chdir('/home/arshdeep/MSOS')
np.save('ex10.npy',np.array(ex10))
np.save('ex11.npy',np.array(ex11))
np.save('ex13.npy',np.array(ex13))
				
np.save('ex14.npy',np.array(ex14))
np.save('ex16.npy',np.array(ex16))
np.save('ex17.npy',np.array(ex17))
								
np.save('ex18.npy',np.array(ex18))
np.save('ex20.npy',np.array(ex20))
np.save('ex21.npy',np.array(ex21))
								
np.save('ex23.npy',np.array(ex23))
np.save('ex24.npy',np.array(ex24))
											
				
np.save('ex_test10.npy',np.array(ex_test10))
np.save('ex_test11.npy',np.array(ex_test11))
np.save('ex_test13.npy',np.array(ex_test13))
				
np.save('ex_test14.npy',np.array(ex_test14))
np.save('ex_test16.npy',np.array(ex_test16))
np.save('ex_test17.npy',np.array(ex_test17))
								
np.save('ex_test18.npy',np.array(ex_test18))
np.save('ex_test20.npy',np.array(ex_test20))
np.save('ex_test21.npy',np.array(ex_test21))
								
np.save('ex_test23.npy',np.array(ex_test23))
np.save('ex_test24.npy',np.array(ex_test24))				
				
np.save('labels',labels)
np.save('labels_test',labels_test)
np.save('dire_list',dirs)				
				
'''	
				
				
#%%				
'''
print('Challenge_data...........................................')
labels_test_chal=[]
ex_test_challenge_4=[]
ex_test_challenge_6=[]
ex_test_challenge_7=[]
ex_test_challenge_8=[];ex_test_challenge_10=[];ex_test_challenge_11=[];ex_test_challenge_13=[];ex_test_challenge_14=[];ex_test_challenge_16=[];ex_test_challenge_17=[];ex_test_challenge_18=[];ex_test_challenge_20=[];ex_test_challenge_21=[];ex_test_challenge_23=[];ex_test_challenge_24=[];
cl_test=0
chal_path="/home/arshdeep/GM_filter_pruning/Fine_tuned_embeddings/DCASE_eva/fold4"
#ds_test=[ex_test_challenge_24]ex_test_challenge_4,ex_test_challenge_6,ex_test_challenge_7,ex_test_challenge_8
#ds_test=[ex_test_challenge_16,ex_test_challenge_17,ex_test_challenge_18,ex_test_challenge_20,ex_test_challenge_21,ex_test_challenge_23,ex_test_challenge_24]
ds_test=[ex_test_challenge_4,ex_test_challenge_6,ex_test_challenge_7,ex_test_challenge_8,ex_test_challenge_10,ex_test_challenge_11,ex_test_challenge_13,ex_test_challenge_14,ex_test_challenge_16,ex_test_challenge_17,ex_test_challenge_18,ex_test_challenge_20,ex_test_challenge_21,ex_test_challenge_23,ex_test_challenge_24]
#ds_test=[ex_test8,ex_test10,ex_test11,ex_test13,ex_test14,ex_test16,ex_test17,ex_test18,ex_test20,ex_test21,ex_test23,ex_test24]
for i in range(np.size(A)):
	j=A[i]
	dict_filename="pruned_layer"+str(j)+"_dict.npy"#"layer"+str(j)+"_acdl.mat" #"pruned_layer"+str(j)+"_dict.npy"##_dict_iter_5.npy"#"pruned_"+/home/arshdeep/DL-COPAR/layer20_dict_iter_10.npy
	layer="layer_"+str(j);
	print(layer)
	labels_test_chal=[]
	cl_test=0
	dict_path=os.path.join(dict_folder,dict_filename)
#	dict_path2=os.path.join("/home/arshdeep/dictOnfeaturemaps/pruned_dict/esc_fold5_0.1",dict_filename)		
	train_path=os.path.join(chal_path,layer)#r,'train_data')
	for root, dirs, files in os.walk(train_path, topdown=False):
		
		for name in dirs:
			if name in dirs:
				parts = []
				parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith('.npy')]
				print(name, "...")
				for part in parts:
					img = np.load(os.path.join(root,name,part))
#					feat=np.matmul(img,np.load(dict_path))#scipy.io.loadmat(dict_path)['D'])
	
#				if np.load(dict_path1).size==16:/media/arshdeep/B294A78494A749A52/DCASE_2019_SOUNDNetfeatures/layer_wise_features
#					feat=np.matmul(img ,np.hstack((np.reshape(np.load(dict_path1),[16,1]),np.load(dict_path2))))
#				else:
#					feat=np.matmul(img ,np.hstack((np.load(dict_path1),np.load(dict_path2))))
#				feat=np.matmul(img ,np.hstack((np.load(dict_path1),np.load(dict_path2))))				

#				example1=np.sum(img,0)
				
#				example_co1=(np.sum(img,0))
#				example_co21=np.sum(np.sum(img[:,0:np.int(np.size(img,1)/2)],0),0)
#				example_co22=np.sum(np.sum(img[:,np.int(np.size(img,1)/2):],0),0)2
#				example_row1=np.sum(np.sum(img,1),0)
#				example_row21=np.sum(np.sum(img[0:np.int(np.size(img,0)/2),:],1),0)
#				example_row22=np.sum(np.sum(img[np.int(np.size(img,0)/2):,:],1),0) 
##				feature=np.hstack((example_co1,example_co21,example_co22,example_row1,example_row21,example_row22));  
#				example2=np.sum(img[0:np.int(np.size(img,0)/2),:],0)
#				example22=np.sum(img[np.int(np.size(img,0)/2):,:],0);
#				example3=np.sum(img[0:np.int(np.size(img,0)/3),:],0);
#				example32=np.sum(img[np.int(np.size(img,0)/3):np.int(np.size(img,0)/3)+np.int(np.size(img,0)/3),:],0);        
#				example33=np.sum(img[np.int(np.size(img,0)/3)+np.int(np.size(img,0)/3):,:],0);
#				example=np.vstack((example1,example2,example22,example3,example32,example33));	
					ds_test[i].append(img)#np.sum(feat,0))#example.flatten())#np.sum(example,0))#np.sum(example,0))#example.flatten())#np.sum(example,0))#,0))#np.sum(img,0))#img.flatten())#np.reshape(img,[img.shape[1],img.shape[0],1]).astype('float'))
					feature=[]
					labels_test_chal.append(cl_test)
					sum += 1
					j=j+1
				cl_test +=1				
'''
#%%		
'''
layer=str(A[0])
data=np.array(ds[0])
print(layer)
filename='sim_index' +layer

index_sorted_CS=[]
#%%validation dataset to select number of filters
class0=data[0:10,:,:]
class1=data[10:20,:,:]
class2=data[140:150,:,:]
class3=data[187:197,:,:]
class4=data[237:247,:,:]
class5=data[291:301,:,:]
class6=data[394:404,:,:]
class7=data[439:449,:,:]
class8=data[469:479,:,:]
class9=data[542:552,:,:]
class10=data[592:602,:,:]
class11=data[681:691,:,:]
class12=data[701:711,:,:]
class13=data[778:788,:,:]
class14=data[860:870,:,:]


validation_data=np.vstack((class0,class1,class2,class3,class4,class5,class6,class7,class8,class9,class10,class11,class12,class13,class14));


sim_mat=[]
data_new=np.reshape(validation_data,(150*np.shape(validation_data)[2],np.shape(validation_data)[1]))
valid_data_normalized=(data_new-np.mean(data_new,0))/np.std(data_new,0)

data_normalized=np.reshape(valid_data_normalized,[150,np.shape(validation_data)[1],np.shape(validation_data)[2]])

data1=data_normalized
#%%
Text
for i in range(150):
    a=data_normalized[i,:,:]
    n = np.linalg.norm(a, axis=0).reshape(1, a.shape[1])
    sim= a.T.dot(a)/n.T.dot(n)
    sim_mat.append(sim[0,:])

sim_mat=np.array(sim_mat).T
sim_std=(np.std(sim_mat,1))


index_sorted_CS=np.argsort(sim_std)

#%%

sim_index=[0]
for i in range(np.size(sim_std)):
    if sim_std[i]>0.006:
        sim_index.append(i)

sim_index=np.array(sim_index)
sim_index8=sim_index
print(np.size(sim_index8))

#%%  Anova method

filter_i=[]
index=[]
data1=data_normalized
pV=[]
for i in range(np.shape(data1)[2]):

    for j in range(150):
        filter_data=data1[j,:,i]
        filter_i.append((filter_data))#,[1,np.size(filter_data)]))
    F,p=stats.f_oneway(filter_i[0],filter_i[1],filter_i[2],filter_i[3],filter_i[4],filter_i[5],filter_i[6],filter_i[7],filter_i[8],filter_i[9],filter_i[10],filter_i[11],filter_i[12],filter_i[13],filter_i[14],filter_i[15],filter_i[16],filter_i[17],filter_i[18],filter_i[19],filter_i[20],filter_i[21],filter_i[22],filter_i[23],filter_i[24],filter_i[25],filter_i[26],filter_i[27],filter_i[28],filter_i[29],filter_i[30],filter_i[31],filter_i[32],filter_i[33],filter_i[34],filter_i[35],filter_i[36],filter_i[37],filter_i[38],filter_i[39],filter_i[40],
				filter_i[41],filter_i[42],filter_i[43],filter_i[44],filter_i[45],filter_i[46],filter_i[47],filter_i[48],filter_i[49],filter_i[50],
				filter_i[51],filter_i[52],filter_i[53],filter_i[54],filter_i[55],filter_i[56],filter_i[57],filter_i[58],filter_i[59],filter_i[60],
				filter_i[61],filter_i[62],filter_i[63],filter_i[64],filter_i[65],filter_i[66],filter_i[67],filter_i[68],filter_i[69],filter_i[70],
				filter_i[71],filter_i[72],filter_i[73],filter_i[74],filter_i[75],filter_i[76],filter_i[77],filter_i[78],filter_i[79],filter_i[80],
				filter_i[81],filter_i[82],filter_i[83],filter_i[84],filter_i[85],filter_i[86],filter_i[87],filter_i[88],filter_i[89],filter_i[90],
				filter_i[91],filter_i[92],filter_i[93],filter_i[94],filter_i[95],filter_i[96],filter_i[97],filter_i[98],filter_i[99],filter_i[100],
				filter_i[101],filter_i[102],filter_i[103],filter_i[104],filter_i[105],filter_i[106],filter_i[107],filter_i[108],filter_i[109],filter_i[110],
				filter_i[111],filter_i[112],filter_i[113],filter_i[114],filter_i[115],filter_i[116],filter_i[117],filter_i[118],filter_i[119],filter_i[120],
				filter_i[121],filter_i[122],filter_i[123],filter_i[124],filter_i[125],filter_i[126],filter_i[127],filter_i[128],filter_i[129],filter_i[130],
				filter_i[131],filter_i[132],filter_i[133],filter_i[134],filter_i[135],filter_i[136],filter_i[137],filter_i[138],filter_i[139],filter_i[140],
				filter_i[141],filter_i[142],filter_i[143],filter_i[144],filter_i[145],filter_i[146],filter_i[147],filter_i[148],filter_i[149])
    print(p,"for",i,"filter")
    pV.append(p)
    if p<0.01 :
print('Challenge_data...........................................')

ex_test_challenge_4=[]
ex_test_challenge_6=[]
ex_test_challenge_7=[]
ex_test_challenge_8=[];ex_test_challenge_10=[];ex_test_challenge_11=[];ex_test_challenge_13=[];ex_test_challenge_14=[];ex_test_challenge_16=[];ex_test_challenge_17=[];ex_test_challenge_18=[];ex_test_challenge_20=[];ex_test_challenge_21=[];ex_test_challenge_23=[];ex_test_challenge_24=[];
cl_test=0
#ds_test=[ex_test_challenge_24]
#ds_test=[ex_test_challenge_16,ex_test_challenge_17,ex_test_challenge_18,ex_test_challenge_20,ex_test_challenge_21,ex_test_challenge_23,ex_test_challenge_24]
ds_test=[ex_test_challenge_4,ex_test_challenge_6,ex_test_challenge_7,ex_test_challenge_8,ex_test_challenge_10,ex_test_challenge_11,ex_test_challenge_13,ex_test_challenge_14,ex_test_challenge_16,ex_test_challenge_17,ex_test_challenge_18,ex_test_challenge_20,ex_test_challenge_21,ex_test_challenge_23,ex_test_challenge_24]
        index.append(i)
    filter_i=[]		

index_sorted_anova=np.argsort(pV)//home/arshdeep/SOoundnet_fold3

#%% Mutual Information....


_row1=np.sum(np.sum(img,1),0)
#				example_row21=np.sum(np.sum(img[0:np.int(np.size(img,0)/2),:],1),0)
#				example_row22=np.sum(np.sum(img[np.int(np.size(img,0)/2):,:],1),0) 
##				feature=np.hstack((example_co1,example_co21,example_co22,example_row1,example_row21,example_row22));  
#				example2=np.sum(img[0:np.int(np.size(img,0)/2),:],0)
#				example22=np.sum(img[np.int(np.size(img,0)/2):,:],0);
#				example3=np.sum(img[0:np.int(np.size(img,0)/3),:],0);
#				example32=np.sum(img[np.int(np.size(img,0)/3):np.int(np.size(img,0)/3)+np.int(np.size(img,0)/3),:],0);        
#				example33=np.sum(img[np.int(np.size(img,0)/3)+np.int(np.size(img,0)/3):,:],0);
#				example=np.vstack((example1,example2,example22,example3,example32,example33));	
					ds_test[i].append(img)#np.sum(example,0))#example.flatten())#np.sum(example,0))#np.sum(example,0))#example.flatten())#np.sum(example,0))#,0))#np.sum(img,0))#img.flatten())#np.reshape(img,[img.shape[1],img.shape[0],1]).astype('float'))
					feature=[]
					labels_test.append(cl_test)
					sum += 1
					j=j+1
				cl_test +=1

filter_i=[]
index=[]
data1=data_normalized
entropy_filter=[]
for i in range(np.shape(data1)[2]):
    for j in range(150):
        filter_data=data1[j,:,i]
        filter_i.append(np.reshape(filter_data,[1,np.size(filter_data)]))
        filter_data=[]
        
#    a=mutual_information((filter_i[0],filter_i[1],filter_i[2],filter_i[3],filter_i[4],filter_i[5],filter_i[6],filter_i[7],filter_i[8],filter_i[9],filter_i[10],filter_i[11],filter_i[12],filter_i[13],filter_i[14],filter_i[15],filter_i[16],filter_i[17],filter_i[18],filter_i[19],filter_i[20],filter_i[31],filter_i[32],filter_i[33],filter_i[34],filter_i[35],filter_i[36],filter_i[37],filter_i[38],filter_i[39],filter_i[40],filter_i[41],filter_i[42],filter_i[43],filter_i[44],filter_i[45],filter_i[46],filter_i[47],filter_i[48],filter_i[49],filter_i[50],filter_i[51],filter_i[52],filter_i[53],filter_i[54],filter_i[55],filter_i[56],filter_i[57],filter_i[58],filter_i[59],filter_i[60],filter_i[61],filter_i[62],filter_i[63],filter_i[64],filter_i[65],filter_i[66],filter_i[67],filter_i[68],filter_i[69],filter_i[70],filter_i[81],filter_i[82],filter_i[83],filter_i[84],filter_i[85],filter_i[86],filter_i[87],filter_i[88],filter_i[89],filter_i[80],filter_i[91],filter_i[92],filter_i[93],filter_i[94],filter_i[95],filter_i[96],filter_i[97],filter_i[98],filter_i[99],filter_i[100],filter_i[101],filter_i[102],filter_i[103],filter_i[104],filter_i[105],filter_i[106],filter_i[107],filter_i[108],filter_i[109],filter_i[110],filter_i[111],filter_i[112],filter_i[113],filter_i[114],filter_i[115],filter_i[116],filter_i[117],filter_i[118],filter_i[119],filter_i[120],filter_i[121],filter_i[122],filter_i[123],filter_i[124],filter_i[125],filter_i[126],filter_i[127],filter_i[128],filter_i[129],filter_i[130],filter_i[131],filter_i[132],filter_i[133],filter_i[134],filter_i[135],filter_i[136],filter_i[137],filter_i[138],filter_i[139],filter_i[140],	filter_i[141],filter_i[142],filter_i[143],filter_i[144],filter_i[145],filter_i[146],filter_i[147],filter_i[148],filter_i[149]), k=1)
     
    a=entropy(np.vstack(filter_i),k=10)			
    entropy_filter.append(a)
#    print(a,"for",i,"filter");
##    a=[]
#    if a==0:d
#					index.append(i)
    filter_i=[]
#    filter_data=[]
index_sorted_DE=np.argsort(entropy_filter) ## increasing order of entropy
#%%

sim_index=index_sorted[2:np.shape(data1)[2]]
sim_index8=sim_index
#a=mutual_information((filter_i[0],filter_i[1],filter_i[2],filter_i[3],filter_i[4],filter_i[5],filter_i[6],filter_i[7],filter_i[8],filter_i[9],filter_i[10],filter_i[11],filter_i[12],filter_i[13],filter_i[14],filter_i[15],filter_i[16],filter_i[17],filter_i[18],filter_i[19],filter_i[20],filter_i[31],filter_i[32],filter_i[33],filter_i[34],filter_i[35],filter_i[36],filter_i[37],filter_i[38],filter_i[39],filter_i[40],filter_i[41],filter_i[42],filter_i[43],filter_i[44],filter_i[45],filter_i[46],filter_i[47],filter_i[48],filter_i[49],filter_i[50],filter_i[51],filter_i[52],filter_i[53],filter_i[54],filter_i[55],filter_i[56],filter_i[57],filter_i[58],filter_i[59],filter_i[60],filter_i[61],filter_i[62],filter_i[63],filter_i[64],filter_i[65],filter_i[66],filter_i[67],filter_i[68],filter_i[69],filter_i[70],filter_i[81],filter_i[82],filter_i[83],filter_i[84],filter_i[85],filter_i[86],filter_i[87],filter_i[88],filter_i[89],filter_i[80],filter_i[91],filter_i[92],filter_i[93],filter_i[94],filter_i[95],filter_i[96],filter_i[97],filter_i[98],filter_i[99],filter_i[100],filter_i[101],filter_i[102],filter_i[103],filter_i[104],filter_i[105],filter_i[106],filter_i[107],filter_i[108],filter_i[109],filter_i[110],filter_i[111],filter_i[112],filter_i[113],filter_i[114],filter_i[115],filter_i[116],filter_i[117],filter_i[118],filter_i[119],filter_i[120],filter_i[121],filter_i[122],filter_i[123],filter_i[124],filter_i[125],filter_i[126],filter_i[127],filter_i[128],filter_i[129],filter_i[130],filter_i[131],filter_i[132],filter_i[133],filter_i[134],filter_i[135],filter_i[136],filter_i[137],filter_i[138],filter_i[139],filter_i[140],	filter_i[141],filter_i[142],filter_i[143],filter_i[144],filter_i[145],filter_i[146],filter_i[147],filter_i[148],filter_i[149]), k=1)


#%%

os.chdir('/home/arshdeep/FILTER_SELECTION_WORK_JULY18/ENTROPY_FOLD1')
np.save(filename,sim_index)




#%% compute distribution of all feature maps and find KL divergence
new_data=data1.flatten()
nanlist=[]
for ii in range(len(new_data)):
   
 if np.isnan(new_data[ii]):
        nanlist.append(ii)
								
				
	#%%						
a,b=np.histogram(data1[:,:,:],bins=15)

distribution_true=a/np.sum(a);
KL=[]

for i in range(np.size(data1,2)):
	range_an=i+1
	a1,b=np.histogram(data1[:,:,index_sorted_CS[0:range_an]],bins=15)
	distrubtion_pred=a1/np.sum(a1)
	KL.append(scipy.stats.entropy(distribution_row1=np.sum(np.sum(img,1),0)
#				example_row21=np.sum(np.sum(img[0:np.int(np.size(img,0)/2),:],1),0)
#				example_row22=np.sum(np.sum(img[np.int(np.size(img,0)/2):,:],1),0) 
##				feature=np.hstack((example_co1,example_co21,example_co22,example_row1,example_row21,example_row22));  
#				example2=np.sum(img[0:np.int(np.size(img,0)/2),:],0)
#				example22=np.sum(img[np.int(np.size(img,0)/2):,:],0);
#				example3=np.sum(img[0:np.int(np.size(img,0)/3),:],0);
#				example32=np.sum(img[np.int(np.size(img,0)/3):np.int(np.size(img,0)/3)+np.int(np.size(img,0)/3),:],0);        
#				example33=np.sum(img[np.int(np.size(img,0)/3)+np.int(np.size(img,0)/3):,:],0);
#				example=np.vstack((example1,example2,example22,example3,example32,example33));	
					ds_test[i].append(img)#np.sum(example,0))#example.flatten())#np.sum(example,0))#np.sum(example,0))#example.flatten())#np.sum(example,0))#,0))#np.sum(img,0))#img.flatten())#np.reshape(img,[img.shape[1],img.shape[0],1]).astype('float'))
					feature=[]
					labels_test.append(cl_test)
					sum += 1
					j=j+1
				cl_test +=1_true,distrubtion_pred, base=None))


plt.plot(KL[0: np.size(data1,2)])


#%%
k=0
for i in range(len(KL)):
	win=KL[i:i+10]
	abs_diff=np.sum(np.abs(np.diff(win)))
	if abs_diff<0.001:
		final_index=i
		print(i)
		k=k+1
		break
		
		
	
/home/arshdeep/making_Sense_dataset
#%%

os.chdir('/home/arshdeep/FILTER_SELECTION_WORK_JULY18/CS_KL_0.01')
sim_index=index_sorted_CS[0:final_index]
np.save(filename,sim_index)
'''
#%%

                    	


#%%


'''
#%% READ LEADERBOARD DATASET
print('Challenge_data...........................................')
A=[8,10,11,13,14,16,17,18,20,21,23,24]
ex_test_challenge_4=[]
ex_test_challenge_6=[]
ex_test_challenge_7=[]
ex_test_challenge_8=[];ex_test_challenge_10=[];ex_test_challenge_11=[];ex_test_challenge_13=[];ex_test_challenge_14=[];ex_test_challenge_16=[];ex_test_challenge_17=[];ex_test_challenge_18=[];ex_test_challenge_20=[];ex_test_challenge_21=[];ex_test_challenge_23=[];ex_test_challenge_24=[];
cl_test=0
#ds_test=[ex_test_challenge_24]
#ds_test=[ex_test_challenge_16,ex_test_challenge_17,ex_test_challenge_18,ex_test_challenge_20,ex_test_challenge_21,ex_test_challenge_23,ex_test_challenge_24]
ds_test=[ex_test_challenge_8,ex_test_challenge_10,ex_test_challenge_11,ex_test_challenge_13,ex_test_challenge_14,ex_test_challenge_16,ex_test_challenge_17,ex_test_challenge_18,ex_test_challenge_20,ex_test_challenge_21,ex_test_challenge_23,ex_test_challenge_24]
for i in range(np.size(A)):
	j=A[i]
	dict_filename="pruned_"+"layer"+str(j)+"_dict.npy"	
	layer="layer_"+str(j);
	print(layer)
	labels_test_chal=[]
	cl_test=0
	dict_path=os.path.join("/home/arshdeep/dictOnfeaturemaps/pruned_dict/esc_fold1_0.1",dict_filename)	
	train_path=os.path.join("/media/arshdeep/B294A78494A749A52/DCASE_2019_SOUNDNetfeatures/layer_wise_leaderboeard",layer)
	for root, dirs, files in os.walk(train_path, topdown=False):
		
		for name in range(0,1200):
			file_name=str(j)+'_'+str(name)+'.npy'
			print(file_name)
			img = np.load(os.path.join(root,file_name))
			ds_test[i].append(np.sum(img,0))
			
'''		
#%%
'''
#			parts = []
#			parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith('.npy')]
			print(name, "...")
			for part in parts:
				img = np.load(os.path.join(root,name,part)) 
#				feat=np.matmul(img,np.load(dict_path))
#				example2=np.sum(img[0:np.int(np.size(img,0)/2),:],0)
#				example22=np.sum(img[np.int(np.size(img,0)/2):,:],0);
#				example3=np.sum(img[0:np.int(np.size(img,0)/3),:],0);
#				example32=np.sum(img[np.int(np.size(img,0)/3):np.int(np.size(img,0)/3)+np.int(np.size(img,0)/3),:],0);        
#				example33=np.sum(img[np.int(np.size(img,0)/3)+np.int(np.size(img,0)/3):,:],0);
#				example=np.vstack((example1,example2,example22,example3,example32,example33));		
				ds_test[i].append(img)#np.sum(example,0))#np.sum(example,0))#example.flatten())#,0))#img.flatten())#np.reshape(img,[img.shape[1],img.shape[0],1]).astype('float'))
				labels_test_chal.append(cl_test)
				sum += 1
				j=j+1
			cl_test +=1
			



#%%
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

os.chdir('/home/arshdeep/journal/scratch_soundnet/fold2_features_scratch')
print("Data Read")
labels=np.load('labels_train.npy')
labels_test=np.load('labels_test.npy')
dirs=np.load('dirs.npy')

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
print(classification_report(labels_test,predd,target_names=dirs))
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
print(classification_report(labels_test,predd,target_names=target_names))


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
print(classification_report(labels_test,predd,target_names=target_names))
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
print(classification_report(labels_test,predd,target_names=target_names))
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
print(classification_report(labels_test,predd,target_names=dirs))
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
print(classification_report(labels_test,predd,target_names=dirs))
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
print(classification_report(labels_test,predd,target_names=dirs))
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
print(classification_report(labels_test,predd,target_names=dirs))
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
print(classification_report(labels_test,predd,target_names=dirs))
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
print(classification_report(labels_test,predd,target_names=dirs))
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
print(classification_report(labels_test,predd,target_names=dirs))

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
print(classification_report(labels_test,predd,target_names=dirs))
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
print(classification_report(labels_test,predd,target_names=dirs))

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
print(classification_report(labels_test,predd,target_names=dirs))
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
print(classification_report(labels_test,predd,target_names=dirs))
#%% maximum a posterioir estimate

#%% weighted probabilites

prob4_pred=weight[0]*prob4_pred
prob6_pred=weight[1]*prob6_pred/home/arshdeep/Soundnet_fold4_data
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

new_prob=prob24_pred+prob23_pred+prob21_pred+prob20_pred+prob18_pred+prob17_pred+prob16_pred+prob14_pred+prob13_pred+prob11_pred+prob10_pred+prob8_pred+prob7_pred+prob4_pred+prob6_pred

#new_prob=1.5*prob21_pred+1.3*prob20_pred+0.45*prob18_pred+1.7*prob17_pred+prob8_pred+prob4_pred+prob7_pred+prob6_pred+prob16_pred+prob14_pred+prob13_pred+0.9*prob11_pred+prob10_pred+1.9*prob24_pred+1.9*prob23_pred

print('after ensemble')
predd=np.argmax(new_prob,1)


target_names = dirs;#['library', 'metro', 'city','car','forest','residen','beach','train','bus','office','park','cafe','tram','home','grocer']
report=(classification_report(labels_test,predd,target_names=dirs))
asd=confusion_matrix(labels_test,predd);
accu=(np.trace(asd)/np.size(labels_test))*100


print(report)
print(accu) 
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
*


#sum4_pred_challenge=clf4.predict(normalize(np.array(ex_test_challenge_4), norm='l2', axis=1, copy=True, return_norm=False))
prob4_pred_c=clf4.predict_proba(normalize( np.array(ex_test_challenge_4), norm='l2', axis=1, copy=True, return_norm=False))

#%% challenge maximum a posteriori
a_c=np.add(prob21_pred_c,prob20_pred_c)
b_c=np.add(prob18_pred_c,prob17_pred_c)
c_c=np.add(a_c,b_c)
d_c=np.add(c_c,prob16_pred_c)
e_c=np.add(prob14_pred_c,prob13_pred_c)me/arshdeep/making_Sense_dataset
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
















