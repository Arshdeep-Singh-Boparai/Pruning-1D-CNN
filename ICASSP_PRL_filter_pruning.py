#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:37:58 2020

@author: arshdeep
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This code ranks the feature maps of SoundNet according to their importance using various criteria for an intermediate layer.
(a) ANOVA-based method (b) Entropy based (c) Angle based deviation (d) Rank or directions based 

The steps are followed as:
	(1) Prepare validation dataset by taking 10 random examples from each class. extract intermediate representations from SoundNet. 
	(2) Apply any of the above mentioned methods (a)-(d) to compute important feature maps.
	(3) Selection of top few important feature maps using mimimum KL-divergence for (a)-(c).
Created on Mon Jul  9 22:45:06 2018

@author: arshdeep
"""

import numpy as np
import scipy.stats as stats
import os
from sklearn.preprocessing import normalize
import math
#%%  Read data of an intermediate layer


layer=str(A[0])
data=np.array(ds[0])  # Read a given layer data
print(layer)
filename='sim_index' +layer

index_sorted_DE=[]
#%% Step 1.  Validation dataset generation


#class0=np.reshape(data[3,:,:],[1,data[0,:,:].shape[0],data[0,:,:].shape[1]])
#class1=np.reshape(data[4,:,:],[1,data[0,:,:].shape[0],data[0,:,:].shape[1]])
#class2=np.reshape(data[7,:,:],[1,data[0,:,:].shape[0],data[0,:,:].shape[1]])
#class3=np.reshape(data[8,:,:],[1,data[0,:,:].shape[0],data[0,:,:].shape[1]])

class0= data[0:10,:,:]
class1=data[57:67,:,:]
class2= data[140:150,:,:]
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

#%%
validation_data=np.vstack((class0,class1,class2,class3,class4,class5,class6,class7,class8,class9,class10,class11,class12,class13,class14));
data_new=np.reshape(validation_data,(150*np.shape(validation_data)[2],np.shape(validation_data)[1]))
valid_data_normalized=(data_new-np.mean(data_new,0))/np.std(data_new,0)
data_normalized=np.reshape(valid_data_normalized,[150,np.shape(validation_data)[1],np.shape(validation_data)[2]])



#%%  (a) Anova-based ordering

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
    #if p<0.01 :
     #   index.append(i)
    filter_i=[]		

index_sorted_anova=np.argsort(pV)

#%% (b) Entropy based ranking


#%%........................................................................................

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
    print(a,"for",i,"filter")
##    a=[]
#    if a==0:
#					index.append(i)
    filter_i=[]
#    filter_data=[]
index_sorted_DE=np.argsort(entropy_filter) ## increasing order of entropy

#%% angular deviation based   ordering........................................................................................
data1=data_normalized
sim_mat=[]
for i in range(150):
    a=data_normalized[i,:,:]
    n = np.linalg.norm(a, axis=0).reshape(1, a.shape[1])
    sim= a.T.dot(a)/n.T.dot(n)
    sim_mat.append(sim[0,:])

sim_mat=np.array(sim_mat).T
sim_std=(np.std(sim_mat,1))

index_sorted_CS=np.argsort(sim_std)


#os.chdir('/home/arshdeep/FILTER_SELECTION_WORK_JULY18/ENTROPY_FOLD1')
#np.save(filename,sim_index)


#%% Selection of top few important feature maps based on minimum distribution  using ordered feature maps obtained from any of the above (a)-(c) methods..........................

#%% compute distribution of all feature maps and find KL divergence...................................................
new_data=data1.flatten()
nanlist=[]
for ii in range(len(new_data)):
    if np.isnan(new_data[ii]):
        nanlist.append(ii)
								
			
	#%%						
a,b=np.histogram(data1[:,:,:],bins=15)

distribution_true=a/np.sum(a);
distribution_true[distribution_true==0]=0.00001
KL=[]

for i in range(np.size(data1,2)):
	range_an=i+1
	a1,b=np.histogram(data1[:,:,index_sorted_CS[0:range_an]],bins=15)
	distrubtion_pred=a1/np.sum(a1)
	distrubtion_pred[distrubtion_pred==0]=0.00001
	
	KL.append(scipy.stats.entropy(distribution_true,distrubtion_pred, base=None))


#%% selection of top few important feature maps using ranked feature maps...............................................................................
k=0
for i in range(len(KL)):
	win=KL[i:i+10]
	abs_diff=np.sum(np.abs(np.diff(win)))
	if abs_diff<0.01:
		final_index=i
		print(i)
		k=k+1
		break

print('The top-l feature maps are :' ,final_index)		
		
	

#%% (d) rank based feature-map selection (geometrical method).................................................................................................................

Th= 150 # threshold 
os.chdir('/home/arshdeep/FILTER_SELECTION_WORK_JULY18/rank_based')
rnk=[]
index=[]
for i in range(np.shape(data_normalized)[2]):
	filter_i=data_normalized[:,:,i]
	rnk.append(np.linalg.matrix_rank(filter_i))
	if rnk[i]==Th:
		index.append(i)


sim_index=index
np.save(filename,np.arra(sim_index)) #save important indexes







