# Pruning-1D-CNN

Definitions:
'''
Redundant feature map: A feature map is redundant if it doesnt provide discriminatory information across various classes.
Important: the feature map which is not redundant.
'''

This code focuses on identifying feature maps which are redundant/irrelavant as defined above in a well-trained Convolution neural network. Three statistical methods [1] and one geometrical method [2] are used to identify redundancy. 



1. Singh, Arshdeep, Padmanabhan Rajan, and Arnav Bhavsar. "Deep hidden analysis: A statistical framework to prune feature maps." ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019.

2. Singh, Arshdeep, Padmanabhan Rajan, and Arnav Bhavsar. "SVD-based redundancy removal in 1-D CNNs for acoustic scene classification." Pattern Recognition Letters 131 (2020): 383-389.
