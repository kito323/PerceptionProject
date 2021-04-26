# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:10:30 2021

@author: Elisa
"""

from report_ex1 import * 
import matplotlib.pyplot as plt
from scipy.linalg import svd

#first normalize the data to avoid any probleme because of scale 

# Subtract the mean from the data
m=X.mean(0)
m1 = np.ones((N, 1))*m
X1 = X - m1

X1 = X1*(1/np.std(X1,0))

X1[:,4] =X[:,4] 

# PCA by computing SVD of X1
U,S,V = svd(X1,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.90

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

#ok so that plot is very different if there is a standardization or not


# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = V.T    

# Project the centered data onto principal component space
Z = X1 @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = plt.figure()
plt.title('data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plt.scatter(Z[class_mask,i], Z[class_mask,j], alpha=.5)
plt.legend(classNames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))

# Output result to screen
plt.show()


pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('PCA Component Coefficients')
plt.show()


