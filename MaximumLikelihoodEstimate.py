import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nrm
import csv
import math
m =10
n = 200 #1x200 data matrix
input_data = []
filepath= raw_input("Enter file path:")
with open(str(filepath))as fin:  #reading csv file
  data = csv.reader(fin)
  for row in data:
        for element in row:
input_data.append(float(element))    

x = np.array(input_data)
print(x)


def exponential(n,x):
    theta = np.mean(x)
    y = nrm.exponential(theta,n)
    plt.hist(x,m)
    plt.hist(y,m,label='exponential')
 
   
def poisson(n,x):
    a = np.mean(x)
    y = nrm.poisson(a,n)
    plt.hist(x,m)
    plt.hist(y,m,label='poisson')
   
   
def gaussian(n,x):
    b = np.std(x)
    a = np.mean(x)
    y = nrm.normal(a,b,n)
    plt.hist(x,m)
    plt.hist(y,m,label='gaussian')
   
def binomial(n,x):
     p = (np.count_nonzero(x==1))*1.0/n
     y = nrm.binomial(n,p,n)
     plt.hist(x,m)
     plt.hist(y,m,label='binomial')
   
   
def laplacian(n,x):
    a = np.mean(x)
    b = np.median(x)
    c = math.sqrt(2)*sum(np.abs(np.array(x)-a)/n)
    y = nrm.laplace(b,c,n)
    plt.hist(x,m)
    plt.hist(y,m,label='laplacian')
   

   
exponential(n,x)
poisson(n,x)
gaussian(n,x)
binomial(n,x)
laplacian(n,x)
plt.legend()
plt.show()
