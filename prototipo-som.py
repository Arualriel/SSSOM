#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 14:24:46 2021

@author: laura
"""

### bibliotecas ###

import numpy as np
import matplotlib.pyplot as plt
import random as rd
from scipy.spatial import Delaunay

### dados ###

n=20 # quantidade de dados

D1=np.zeros((2,n))
D2=np.zeros((2,n))
D3=np.zeros((2,n))

#rd.seed(1)

for i in range(n):
    D1[0,i]=rd.normalvariate(0,0.25)
    D2[0,i]=rd.normalvariate(0,0.25)
    D3[0,i]=rd.normalvariate(5,0.25)

for i in range(n):
    D1[1,i]=rd.normalvariate(5,0.25)
    D2[1,i]=rd.normalvariate(0,0.25)
    D3[1,i]=rd.normalvariate(0,0.25)
    
    
plt.title("Dados")
plt.scatter(D1[0,:],D1[1,:],c='b',label='Dados 1')
plt.scatter(D2[0,:],D2[1,:],c='r',label='Dados 2')
plt.scatter(D3[0,:],D3[1,:],c='g',label='Dados 3')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

### som ###

l=25 # numero de nos na rede

W=np.zeros((2,l))

for i in range(2):
    for j in range(l):
        W[i,j]=rd.normalvariate(2.5,0.5)

tri = Delaunay(W.T)

plt.title("SOM")
plt.scatter(D1[0,:],D1[1,:],c='b',label='Dados 1')
plt.scatter(D2[0,:],D2[1,:],c='r',label='Dados 2')
plt.scatter(D3[0,:],D3[1,:],c='g',label='Dados 3')
plt.triplot(W[0,:], W[1,:], tri.simplices.copy(),c='k')
plt.scatter(W[0,:],W[1,:],c='k',label='Pesos')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

eta0=0.1
tau2=1000.0
sigma0=2.5
tau1=1000.0/np.log(sigma0)
eta=eta0
sigma=sigma0
T=100 # numero de iteracoes
x=np.zeros(2)
for t in range(0,T+1):
    k=rd.randint(0,3*n-1)
    if((k>=0) and (k<n)):
        x=D1[:,k]
    elif((k>=n) and (k<2*n)):
        x=D2[:,k-n]
    else:
        x=D3[:,k-2*n]
    
    d=100000
    i=0
    for j in range(l):
        dj=((x[0]-W[0,j])**2.0+(x[1]-W[1,j])**2.0)**0.5
        if (dj<=d):
            d=dj
            i=j
         
    for j in range(l):
        dw=((W[0,i]-W[0,j])**2.0+(W[1,i]-W[1,j])**2.0)**0.5
        h=np.exp((-dw**2.0)/(2.0*sigma**2.0))
        if (dw<h):
            W[0,j]=W[0,j]+eta*(x[0]-W[0,j])
            W[1,j]=W[1,j]+eta*(x[1]-W[1,j])
        
            plt.title("SOM")
            plt.scatter(D1[0,:],D1[1,:],c='b',label='Dados 1')
            plt.scatter(D2[0,:],D2[1,:],c='r',label='Dados 2')
            plt.scatter(D3[0,:],D3[1,:],c='g',label='Dados 3')
            plt.scatter(x[0],x[1],c='cyan',label='P(t)')
            plt.triplot(W[0,:], W[1,:], tri.simplices.copy(),c='k')
            plt.scatter(W[0,:],W[1,:],c='k',label='Pesos')
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()
    
    eta=eta0*np.exp(-t/tau2)
    sigma=sigma0*np.exp(-t/tau1)




        