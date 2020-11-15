import numpy as np

# Input Parameters

Hidden_nodes = int(input("Number of Hidden Layer nodes :"))
N = int(input("Number of Training Samples :"))
std_dev = float(input("Standard deviation of Noise:"))
epochs = int(input("Number of epochs: "))
lr = float(input("Enter learning rate:"))
opt = int(input("Enter the corresponding number of operation:\n 1) XOR\n 2) OR\n 3)AND\n"))

def Operation(opt):
    if opt==1:
       return np.array([0,1,1,0])

    if opt==2:
       return np.array([0,1,1,1])

    if opt==3:
       return np.array([0,0,0,1])   
  
  
X = []
Y = []

Input =  np.array([[0,0],[0,1],[1,0],[1,1]])
Output = Operation(opt)
M = len(Input)
  
for i in range(M):
    for j in range(N):
        X.append(Input[i]+np.random.normal(0,std_dev,(1,2)))
        Y.append(Output[i]+np.random.normal(0,std_dev))

X = np.array(X)
X = X.reshape((len(Y),2))
Y = np.array(Y)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def Dsigmoid(x):
    return sigmoid(x)*sigmoid(1-x)

def weights(Hidden_nodes):  
    W1 = np.random.normal(0,1,(2,Hidden_nodes))
    B1 = np.random.normal(0,1,(Hidden_nodes,1))
    W2 = np.random.normal(0,1,(Hidden_nodes,1))
    B2 = np.random.normal(0,1,(1,1))
    return W1,W2,B1,B2

def layer(x,W,b):
    return np.matmul(W.T,x.reshape(len(x),1)) + b

def forwardProp(W1,W2,B1,B2,x):
    z=sigmoid(layer(x,W1,B1))
    y_pred=sigmoid(layer(z,W2,B2))
    return z,y_pred

def mse(TrueY,PredY):
    return (TrueY-PredY)**2/len(Y)
    



#Testing the trained dataset
W1,W2,B1,B2 = weights(Hidden_nodes)
Loss=0
for i in range(epochs):
    delta1 = np.zeros(W1.shape)
    delta2 = np.zeros(W2.shape)
    bias1 = np.zeros(B1.shape)
    bias2 = np.zeros(B2.shape)
    #BackPropagation
    z,y_pred = forwardProp(W1,W2,B1,B2,X[i])
    layer_1 = layer(X[i],W1,B1)
    layer_2 = layer(z,W2,B2)
    bias2 += 2*(y_pred-Y[i])*Dsigmoid(layer_2)
    delta2 += 2*(y_pred-Y[i])*Dsigmoid(layer_1)*z
    Loss =  Loss+mse(Y[i],y_pred)
    for j in range(Hidden_nodes):
        
        bias1[j] -= (2*(y_pred-Y[i])*Dsigmoid(layer_2)*Dsigmoid(layer_1[j])*W2[j]).reshape(1,)
        delta1[:,j] -= (2*(y_pred-Y[i])*Dsigmoid(layer_2)*Dsigmoid(layer_1[j])*W2[j]*X[i]).reshape(2,)
        print(Loss)

def TestTrain():
    while(1):
        sample_test = np.array((2,1),dtype='f')
        sample_test[0]=float(input())
        sample_test[1]=float(input())
        print(sample_test)
        z,y_pred = forwardProp(W1,W2,B1,B2,sample_test)
        if (y_pred>0.5):
           print("1")
        if (y_pred<0.5):
           print("0")
    

    
B1-=lr*bias1
B2-=lr*bias2
W2-=lr*delta2
W1-=lr*delta1

TestTrain()
