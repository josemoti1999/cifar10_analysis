import numpy as np
X=np.array([[0.8],[0.7],[0.1]])
print(X.shape)
W1=np.array([[0.1861,0.2878,0.3983],[-0.1954,0.10403,0.500575]])
print(W1.shape)
b1=np.array([[-0.21733],[-0.1942]])
print(b1.shape)
W2=np.array([[-0.1709,0.20292]])
b2=np.array([[0.03376]])

def sigmoid(x):
    a=1/(1+np.exp(-x))
    return a
    


Z1=np.dot(W1,X)+b1
print(Z1)
A1=sigmoid(Z1)
print(A1)
Z2=np.dot(W2,A1)+b2
print(Z2)
A2=sigmoid(Z2)
print(A2)