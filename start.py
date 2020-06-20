#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import cvxopt
import cvxopt.solvers
from collections import Counter
from itertools import combinations_with_replacement
from time import time


# In[ ]:


# read the files
xtr0 = pd.read_csv('Xtr0.csv', " ", header=0)
xtr1 = pd.read_csv('Xtr1.csv', " ", header=0)
xtr2 = pd.read_csv('Xtr2.csv', " ", header=0)
xtrain_all_files = np.append(np.append(xtr0, xtr1), xtr2)
X_train = np.array(xtrain_all_files)

xte0 = pd.read_csv('Xte0.csv', " ", header=0)
xte1 = pd.read_csv('Xte1.csv', " ", header=0)
xte2 = pd.read_csv('Xte2.csv', " ", header=0)
xtest_all_files = np.append(np.append(xte0, xte1), xte2)
X_test = np.array(xtest_all_files)

ytr0 = pd.read_csv('Ytr0.csv', index_col=0, header=0)
ytr1 = pd.read_csv('Ytr1.csv', index_col=0, header=0)
ytr2 = pd.read_csv('Ytr2.csv', index_col=0, header=0)
ytrain_all_files = np.append(np.append(ytr0, ytr1), ytr2)
Y_train = np.array(ytrain_all_files)
Y_train[Y_train[:] == 0] = -1


# In[4]:


# k-mers function
def func_create_subsequences(length):
    p = ['A','C','G','T','C','G','T','A','G','T','A','C','T','A','C','G', 'A','C','G','T']
    sub_sequence = []
    for i in combinations_with_replacement(p, length):
        sub_sequence.append(list(i))
    sub_seq = np.asarray(sub_sequence)    
    sub_sequence= np.unique(sub_sequence, axis = 0) 
    sub_sequence =["".join(j) for j in sub_sequence[:,:].astype(str)]
    print('def subseq ok')
    return sub_sequence
def feature_extraction(x, subsequence, index, k):
    features = np.zeros((len(x), len(subsequence)))   #To store the occurence of each string
    for i in range(0,len(x)):
        s = x[i]
        c = [ s[j:j+k] for j in range(len(s)-k+1) ]
        counter = Counter(c)
        j=0
        for m in subsequence:
            features[i][j] = counter[m]
            j=j+1
    features_train = features[:,index]
    features_train = features_train / np.max(np.abs(features_train),axis=0)
    print('def feat extr ok')

    return features_train


# In[5]:


# SVM clasifier with kernels

def kernel_poly(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def kernel_sigmoid(x, y, gamma = 0.02): # with alternative representation of parameter Sigma
    '''
    Parameters:
    x - sample 1
    y - sample 2
    gamma = 1/2sigma^2, where sigma^2 is variance. Choose gamma > 0 
    ''' 
    return np.exp(-gamma*(np.linalg.norm(x-y))**2)
                  
def kernel_linear(x, y):
    '''
    Parameters:
    x - sample 1
    y - sample 2
    '''              
    return np.dot(x, y)


def kernel_sigmoid(x, y, n_features = 4056, theta = -1.4):
    '''
    Parameters:
    x - sample 1
    y - sample 2
    theta - free parameter, <0 
    kernel also uses normalization parameter (1/n_features)
    '''
    # best benachmark theta from Smola book chapt 7.8.1 is -1.4
    return np.tanh((1/n_features)*np.dot(x,y)+(theta)) 

class SVM(object):
    def __init__(self, kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        print(X)
        n_samples, n_features = X.shape

        K = np.zeros((n_samples, n_samples)) # kernel matrix
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])
        
        # input structures for CVXOPT

        P = cvxopt.matrix(np.outer(y,y) * K) 
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples),'d')
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))


        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        a = np.ravel(solution['x'])


        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
      
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        if self.kernel == kernel_linear:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None



    def predict(self, X):
        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
            y_pred[i] = s
        return np.sign(y_pred + self.b)


# In[8]:


# feature extraction
k=6
sub_sequence = func_create_subsequences(k)
index = np.arange(0, len(sub_sequence))
x_train_feat = feature_extraction(X_train, sub_sequence, index, k)
x_test_feat = feature_extraction(X_test,sub_sequence, index, k)


# In[ ]:


# fitting and prediction
clf = SVM(kernel_linear, 0.1)
clf.fit(x_train_feat, Y_train) 
y_pred = clf.predict(x_test_feat)


# In[ ]:


len(y_pred)
inds = [i for i in range(0,3000)]
Ypred = pd.DataFrame(columns=['Id','Bound'])
Ypred['Bound'] = y_pred
Ypred['Id'] = inds
Ypred.head()
Ypred.to_csv('Yte.csv', index=False)
y_pred[y_pred[:] == -1] = 0
y_pred = y_pred.astype(int)

