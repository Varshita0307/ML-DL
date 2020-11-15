import numpy as np
import numpy.linalg as la
import matplotlib.pylpot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))

def Dsigmoid(x):
    return sigmoid(x)*sigmoid(1-x)

def layer(x,W,b):
    return np.matmul(x,W)+b
    
def weights(Hidden_nodes, input_size):
    W1 = np.random.normal(0,1,(input_size,Hidden_nodes))
    B1 = np.random.normal(0,1,(1,Hidden_nodes))
    W2 = np.random.normal(0,1,(Hidden_nodes,input_size))
    B2 = np.random.normal(0,1,(1,input_size))
    return W1,W2,B1,B2
    
def fowardProp(W1,W2,B1,B2,x):
    z=sigmoid(layer(x,W1,B1))
    y=sigmoid(layer(z,W2,B2))
    return z,y

def mse(TrueY,PredY):
    return (TrueY-PredY)**2
    
def loss(data):
        z,y= forwardProp(W1,W2,B1,B2,data)
        error = np.sum((data - y)**2)
        zm = np.mean(z, axis=0)
        regularizer = Lambda*np.sum(p*np.log(p/zm) + (1-p)*np.log((1-p)/(1-zm)))
        return err+regularizer
    
    
data_train = np.load("reshaped_compressed_MNIST.npy")
Hidden_nodes = 225
input_size = 196
epochs = 50
lr = 1e-4
samples = 60000
data = data_train.reshape(samples,input_size)
data = data/255.0
regularization = 1.0
s = 0.1
Lambda = 1.0

W1,W2,B1,B2 = weights(Hidden_nodes,input_size)
Loss = 0
for i in range(epochs):
	delta1 = np.zeros(W1.shape)
    delta2 = np.zeros(W2.shape)
    bias1 = np.zeros(B1.shape)
    bias2 = np.zeros(B2.shape)
    z,y = forwardProp(W1,W2,B1,B2,data)
    layer_1 = layer(data,W1,B1)
    layer_2 = layer(z,W2,B2)
    zm = np.mean(z,axis = 0)
    dz = z*(1-z)
    y_del = 2*(y-data) *Dsigmoid(layer_2)
    z_del = -dz * np.matmul(y_del, W2) 
    dK = Lambda * ((s/zm) + ((1-s)/(1-zm))) * dZ
    delta1 = np.matmul((z_del + dK).T , data)
    delta2 = np.matmul((y_del).T , data)
    bias1 = np.sum((z_del+dK), axis=0)
    bias2 = np.sum(y_del, axis=0)
    
 W1 -=lr*delta1
 W2 -=lr*delta2
 B1 -=lr*bias1
 B2 -=lr*bias2
    
    

    
    
    
    
    

	
	
    
  
