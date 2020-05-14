import numpy as np
from numpy import linalg

np.random.seed(0)

#generating data

n = int(input("Number of data samples:"))
d = int(input("Dimension of samples:"))
k = int(input("Gaussian mode:"))
threshold=float(input("threshold value: "))
iterations = int(input("No. of iterations :"))


def data(n,d):
    mean=np.linspace(0,1,d)
    covar=np.identity(d)
    x=np.random.multivariate_normal(mean,covar,n)
    print(x.shape)
    return x
    
def multigauss(mean,covar,x):
    mean=mean.reshape(1,len(mean))
    nr = np.exp(-0.5 * ((x - mean.T).T * np.linalg.inv(covar) * (x - mean.T)))
    dr = np.sqrt(((2*np.pi)**mean.shape[0]) * np.linalg.det(covar))
    pdf = nr/dr
    return pdf
    
def gam(mean,covar,x,weights):
    tot = []
    gamma = []
    for j in range (len(weights)):
        temp = weights[j]*multigauss(mean[:,j],covar[:,j],x)
        tot.append(temp)
    tot = np.array(tot)
    gamma = tot/np.sum(tot)
    return tot,gamma
    
def log_likelihood(mean,covar,x,weights,k,n):
    a =[]
    Lx=0
    for i in range(k):
        temp=0
        for j in range(n):
            temp =(np.sum(weights[i]*multigauss(mean[:,j],covar[:,j],x)))
            a.append(temp)
    Lx += np.sum([np.log(temp)])
    return Lx

def updated_parameters(mean,x,gamma,covar):
    Nk=np.sum(gamma,axis=0)
    Nk=Nk.reshape(len(Nk),1)   
    w_new = Nk/(len(gamma)*1.0)
    mean=np.dot(gamma.T,X)
    mean=mean.reshape(gamma.shape[1],X.shape[1])
    mean_new = mean/Nk
    for i in range(gamma.shape[1]):
        temp=(gamma.T)[i]
        temp=temp.reshape(len(temp),1)
        mean_temp=mean[i]
        mean_temp=mean_temp.reshape(1,len(mean_temp))
        Nk=np.sum(temp)
        covar_new[i]=(np.matmul((temp*(X-mean_temp)).T,X-mean_temp))/Nk

    return covar_new, mean_new, w_new

def GMM(x,k,n,d,threshold):
    count = 0
    mean = np.random.rand(k,d)
    covar = np.random.uniform(0,1,(k,d,d))
    weights = (1.0/k)*np.ones(k)
    initialL= log_likelihood(mean,covar,x,weights,k,n)
 
    while(count<iterations):
        for i in range(n):
            temp,_=gam(mean,covar,x[i],weights)
            gamma.append(temp)


    gamma=np.array(gamma)
    gamma=gamma.reshape(n,k)
    NewCovar = updated_parameters(mean,x,gamma,covar)
    NewMean = updated_parameters(mean,x,gamma,covar)
    NewW = updated_parameters(mean,x,gamma,covar)

    finalL = log_likelihood(NewMean,NewCovar,x,NewW,k,n)
    error = finalL-initialL
    print('error\n'+str(error))
    count+=1

        
    print('NewCovar\n'+str(NewCovar))
    print('NewMean\n'+str(NewMean))
    print('NewW\n'+str(NewW))


x = data(n,d)
GMM(x,k,n,d,threshold)
