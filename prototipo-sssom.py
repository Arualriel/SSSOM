
#!/usr/bin/env pythonm-1
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 20:37:11 2021

@author: laura
"""


### bibliotecas ###

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random as rd



### dados ###


n=10 # quantidade de dados em cada cluster
m=4 # dimensao + 1

D1=np.zeros((m,n))
D2=np.zeros((m,n))
D3=np.zeros((m,n))
#rd.seed(1)

D1[m-1,:]=999
D2[m-1,:]=999
D3[m-1,:]=999

rd.seed(11)

for i in range(n):
    D1[0,i]=rd.normalvariate(1.0,0.07)
    D2[0,i]=rd.normalvariate(0.5,0.07)
    D3[0,i]=0.0 #rd.normalvariate(5,0.25)
    D1[1,i]=rd.normalvariate(0.5,0.07)
    D2[1,i]=0.0 #rd.normalvariate(0,0.25)
    D3[1,i]=rd.normalvariate(1.0,0.07)
    D1[2,i]=0.0#rd.normalvariate(5,0.25)
    D2[2,i]=rd.normalvariate(1.0,0.07)
    D3[2,i]=rd.normalvariate(0.5,0.07)


D1[m-1,n-int(n/3):n+1]=1
D2[m-1,n-int(n/3):n+1]=2
D3[m-1,n-int(n/3):n+1]=3

    
### grafico ###
 
     
fig0 = plt.figure(figsize=(10, 10))
ax0 = plt.axes(projection="3d")
   
ax0.scatter3D(D1[0,:],D1[1,:],D1[2,:], c='red',alpha=1)
ax0.scatter3D(D2[0,:],D2[1,:],D2[2,:], c='green',alpha=1)
ax0.scatter3D(D3[0,:],D3[1,:],D3[2,:], c='blue', alpha=1)
ax0.set_title('Dados', fontsize=18)
ax0.set_xlabel('X', fontsize=15)
ax0.set_ylabel('Y', fontsize=15)
ax0.set_zlabel('Z', fontsize=15)# [view_init] Modifica o ângulo de visualização do gráfico
ax0.view_init(50, 35)
plt.show()
   
### self-organization phase ###

S=n*3
at=0.9
eb=0.0005
en=0.002*eb
beta=0.01
c=0.5
minwd=0.25
Nmax=25
maxcomp=2*S
lp=0.005
tmax=10*S
k=0
N=0
push_rate=0.05*eb 
C=np.zeros((m,Nmax))
delta=np.zeros((m-1,Nmax))
W=np.zeros((m-1,Nmax))
noclass=999
Class=np.zeros(Nmax)
wins=np.zeros(Nmax)
nwins=1
age_wins=S
conexao=np.zeros((Nmax,Nmax))
for i in range(Nmax):
    conexao[i,i]=1
epsilon=0.01
k=0
s= 0.05 ## no intervalo [0.01,0.1]
j=0

k=rd.randint(0,3*n-1)
if((k>=0) and (k<n)):
    x=D1[:,k]
elif((k>=n) and (k<2*n)):
    x=D2[:,k-n]
else:
    x=D3[:,k-2*n]
   

C[:,0]=x[:]
N=1
if((x[m-1]!=noclass) and (x[m-1]!=0)):
    classj=x[m-1]
elif(x[m-1]!=0):
    classj=noclass

for t in range(tmax+1):
   k=rd.randint(0,3*n-1)
   if((k>=0) and (k<n)):
       x=D1[:,k]
   elif((k>=n) and (k<2*n)):
       x=D2[:,k-n]
   else:
       x=D3[:,k-2*n]
   
   if(t==0):
       C[:,0]=x[:]
       N=2
   a_s=0
   cont=0
   j=0
   ind=0
   for i in range(Nmax):
      if(C[m-1,i]!=0):
         cont=cont+1
         somaw=0.0
         for r in range(m-1):
            somaw=somaw+W[r,i]
         Dw=0.0
         for r in range(m-1):
            Dw=Dw+W[r,i]*(x[r]-C[r,i])**2.0
         Dw=(Dw)**0.5
         ac=somaw/(somaw+Dw+epsilon)
         #### max (activation) ###
         if (ac>=a_s):
            a_s=ac
            ind=i
      elif((j==0) and (C[m-1,i]==0)):
         j=i
         
   s1=C[:,ind]
   N=cont
   
   #### supervised mode ####
   if((x[m-1]!=noclass) and (x[m-1]!=0)):
      if((s1[m-1]==noclass)or(s1[m-1]!=0)):
         if ((a_s<at) and (N<Nmax)):
            C[:,j]=x[:]
            W[:,j]=1.0
            wins[j]=0
            classj=x[m-1]
            for l in range(Nmax):
               for r in range(Nmax):
                  normaw=np.linalg.norm(W[:,r]-W[:,l])
                  classr=C[m-1,r]
                  if (((((classr==classj) or (classr==noclass)) or(classj==noclass)) and (normaw<c*(m**0.5))) and (classr!=0)):
                     conexao[r,l]=1
                  else:
                     conexao[r,l]=0
         elif(a_s>=at):
            #### update node ###
            for l in range(Nmax):
               if (conexao[l,ind]!=0):
                  if (l!=ind):
                      e=en
                  else:
                      e=eb
                  deltajmin=10000000
                  deltajmax=0
                  deltajmean=0
                  for r in range(m-1):
                     delta[r,l]=(1-e*beta)*delta[r,l]+e*beta*(np.abs(x[r]-C[r,l]))
                     if (delta[r,l]<=deltajmin):
                         deltajmin=delta[r,l]
                     if(delta[r,l]>=deltajmax):
                        deltajmax=delta[r,l]
                     deltajmean=deltajmean+delta[r,l]/(m-1)
             
             
                  if(deltajmin!=deltajmax):
                     for r in range(m-1):
                        W[r,l]=1.0/(1.0+np.exp((deltajmean-delta[r,l])/(s*(deltajmax-deltajmin))))
                  else:
                     W[:,l]=1.0
            
                  for r in range(m-1):
                     C[r,l]=C[r,l]+e*(x[r]-C[r,l])
           
            s1[m-1]=x[m-1]  
            classj=s1[m-1]
            for r in range(Nmax):
               normaw=np.linalg.norm(W[:,r]-W[:,ind])
               classr=C[m-1,r]
               if (((((classr==classj) or (classr==noclass)) or(classj==noclass)) and (normaw<c*(m**0.5))) and (classr!=0)):

                  conexao[r,ind]=1
               else:
                  conexao[r,ind]=0
                   
            wins[ind]=wins[ind]+1
        
      else:   
         s2=np.zeros(m)
         cont=0
         j=0
         for z in range(Nmax):
            if((C[m-1,z]!=0) or (C[m-1,z]==x[m-1])):
               cont=cont+1
               somaw=0.0
               for r in range(m-1):
                  somaw=somaw+W[r,z]
               Dw=0.0
               for r in range(m-1):
                  Dw=Dw+W[r,z]*(x[r]-C[r,z])**2.0
               Dw=(Dw)**0.5
               ac=somaw/(somaw+Dw+epsilon)
               #### max (activation) ###
               if (ac>a_s):
                  a_s=ac
                  ind2=z
            elif((j==0) and (C[m-1,z]==0)):
               j=z
         N=cont
         if ((C[:,ind2]!=s1) and (a_s>=at)):
            s2=C[:,ind2]
         if (s2[m-1]!=0):
            for l in range(Nmax):
               if (conexao[l,ind2]!=0):
                  if (l!=ind2):
                     e=en
                  elif(l==ind):
                     e=push_rate
                  else:
                     e=eb
                  deltajmin=10000000
                  deltajmax=0
                  deltajmean=0
                  for r in range(m-1):
                     delta[r,l]=(1-e*beta)*delta[r,l]+e*beta*(np.abs(x[r]-C[r,l]))
                     if (delta[r,l]<=deltajmin):
                        deltajmin=delta[r,l]
                     if(delta[r,l]>=deltajmax):
                        deltajmax=delta[r,l]
                     deltajmean=deltajmean+delta[r,l]/(m-1)
             
             
                  if(deltajmin!=deltajmax):
                     for r in range(m-1):
                        W[r,l]=1.0/(1.0+np.exp((deltajmean-delta[r,l])/(s*(deltajmax-deltajmin))))
                  else:
                     W[:,l]=1.0
            
                  for r in range(m-1):
                     C[r,l]=C[r,l]+e*(x[r]-C[r,l])
                 
               elif(l==ind):
                  e=push_rate
                   
                  deltajmin=10000000
                  deltajmax=0
                  deltajmean=0
                  for r in range(m-1):
                     delta[r,l]=(1-e*beta)*delta[r,l]+e*beta*(np.abs(x[r]-C[r,l]))
                     if (delta[r,l]<=deltajmin):
                        deltajmin=delta[r,l]
                     if(delta[r,l]>=deltajmax):
                        deltajmax=delta[r,l]
                     deltajmean=deltajmean+delta[r,l]/(m-1)
             
             
                  if(deltajmin!=deltajmax):
                     for r in range(m-1):
                        W[r,l]=1.0/(1.0+np.exp((deltajmean-delta[r,l])/(s*(deltajmax-deltajmin))))
                  else:
                        W[:,l]=1.0
           
                  for r in range(m-1):
                     C[r,l]=C[r,l]+e*(x[r]-C[r,l])
            wins[ind2]=wins[ind2]+1       
         elif(N<Nmax):
            C[:,j]=x[:]
            W[:,j]=1.0
            wins[j]=0
            classj=x[m-1]
            for r in range(Nmax):
               normaw=np.linalg.norm(W[:,r]-W[:,j])
               classr=C[m-1,r]
               if (((((classr==classj) or (classr==noclass)) or(classj==noclass)) and (normaw<c*(m**0.5))) and (classr!=0)):

                  conexao[r,j]=1
               else:
                  conexao[r,j]=0 
   else:
      #### unsupervised mode ####
      cont=0
      j=0
      for i in range(Nmax):
         if(C[m-1,i]!=0):
            cont=cont+1
         elif((j==0) and (C[m-1,i]==0)):
            j=i
      N=cont
      if((a_s<at) and (N<Nmax)):
         C[:,j]=x[:]
         W[:,j]=1.0
         wins[j]=0
         classj=x[m-1]
         for r in range(Nmax):
            normaw=np.linalg.norm(W[:,r]-W[:,j])
            classr=C[m-1,r]
            if (((((classr==classj) or (classr==noclass)) or(classj==noclass)) and (normaw<c*(m**0.5))) and (classr!=0)):

               conexao[r,j]=1
            else:
               conexao[r,j]=0
      else:
         #### update node ###
         for l in range(Nmax):
            if (conexao[l,ind]!=0):
               if (l!=ind):
                   e=en
               else:
                   e=eb
               deltajmin=10000000
               deltajmax=0
               deltajmean=0
               for r in range(m-1):
                  delta[r,l]=(1-e*beta)*delta[r,l]+e*beta*(np.abs(x[r]-C[r,l]))
                  if (delta[r,l]<=deltajmin):
                     deltajmin=delta[r,l]
                  if(delta[r,l]>=deltajmax):
                     deltajmax=delta[r,l]
                  deltajmean=deltajmean+delta[r,l]/(m-1)
             
             
               if(deltajmin!=deltajmax):
                  for r in range(m-1):
                     W[r,l]=1.0/(1.0+np.exp((deltajmean-delta[r,l])/(s*(deltajmax-deltajmin))))
               else:
                     W[:,l]=1.0
          
               for r in range(m-1):
                  C[r,l]=C[r,l]+e*(x[r]-C[r,l])
         wins[ind]=wins[ind]+1   
   if(nwins==age_wins):
      
      ### remove nodes ###            
      for r in range(Nmax):         
         if(wins[r]<(lp*age_wins)):
            conexao[r,:]=0
            conexao[:,r]=0
            C[:,r]=0
         wins[r]=0
      nwins=0
   nwins=nwins+1
   
   print(wins)
   
   
   
   ### grafico ###

   fig0 = plt.figure(figsize=(10, 10))
   ax0 = plt.axes(projection="3d")
    
   ax0.scatter3D(D1[0,:],D1[1,:],D1[2,:], c='red', alpha=1)
   ax0.scatter3D(D2[0,:],D2[1,:],D2[2,:], c='green', alpha=1)
   ax0.scatter3D(D3[0,:],D3[1,:],D3[2,:], c='blue', alpha=1)
   ax0.set_title('Dados', fontsize=18)
   # [view_init] Modifica o ângulo de visualização do gráfico
   #ax0.view_init(50, 35)
   #plt.show()
   i=0
   cont=0
   for r in range(Nmax):
      if(C[m-1,r]!=0):
         cont=cont+1
   Nos=np.zeros((m-1,cont))
   for r in range(Nmax):
      if(C[m-1,r]!=0):
         Nos[0,i]=C[0,r]
         Nos[1,i]=C[1,r]
         Nos[2,i]=C[2,r]
         i=i+1       
   for r in range(Nmax):
      for g in range(Nmax):
         if((((C[m-1,r]!=0) and (C[m-1,g]!=0)) and (g!=r)) and (conexao[r,g]==1)):
            xplot=[C[0,r],C[0,g]]
            yplot=[C[1,r],C[1,g]]
            zplot=[C[2,r],C[2,g]]
            ax0.plot3D(xplot,yplot,zplot,c='black')
   X=Nos[0,:]
   Y=Nos[1,:]
   Z=Nos[2,:]
   ax0.scatter3D(X, Y, Z, c='black',s=70, alpha=1)
   ax0.set_title('SSSOM', fontsize=18)
  
   ax0.set_xlabel('X', fontsize=15)
   ax0.set_ylabel('Y', fontsize=15)
   ax0.set_zlabel('Z', fontsize=15)# [view_init] Modifica o ângulo de visualização do gráfico
   ax0.view_init(50, 35)
   plt.show()
print(wins)        
#### Convergence phase ####

remocoes=1  
nmax=0  
while (remocoes!=0):    
   cont=0
   remocoes=0
   for i in range(Nmax):
      if(C[m-1,i]!=0):
          cont=cont+1
   nmax=cont
   #Nmax=N
   ### remove nodes ###
   print(wins)            
   for r in range(Nmax):       
      
      if(wins[r]<(lp*age_wins)):
         remocoes=1
         conexao[r,:]=0
         conexao[:,r]=0
         C[:,r]=0
   cont=0
   for i in range(Nmax):
      if(C[m-1,i]!=0):
          cont=cont+1
   N=cont
   if((N==nmax) or (N==1)):
      remocoes=0
      classj=x[m-1]
   print(remocoes)
   if(remocoes!=0):
      for r in range(Nmax):
         for v in range(Nmax): 
            normaw=np.linalg.norm(W[:,r]-W[:,v])
            classr=C[m-1,r]
            if (((((classr==classj) or (classr==noclass)) or(classj==noclass)) and (normaw<c*(m**0.5))) and (classr!=0)):
               conexao[r,v]=1
               print(C[0:m-2,r],x[0:m-2])
            else:
               conexao[r,v]=0
      wins[:]=0
      for t in range(tmax):
         k=rd.randint(0,3*n-1)
         if((k>=0) and (k<n)):
             x=D1[:,k]
         elif((k>=n) and (k<2*n)):
             x=D2[:,k-n]
         else:
             x=D3[:,k-2*n]
   
         for i in range(Nmax):
            if(C[m-1,i]!=0):
               cont=cont+1
               somaw=0.0
               for r in range(m-1):
                  somaw=somaw+W[r,i]
               Dw=0.0
               for r in range(m-1):
                  Dw=Dw+W[r,i]*(x[r]-C[r,i])**2.0
               Dw=(Dw)**0.5
               ac=somaw/(somaw+Dw+epsilon)
               #### max (activation) ###
               if (ac>a_s):
                  a_s=ac
                  ind=i   
                     
         #### update node ###
         for l in range(Nmax):
            if (conexao[l,ind]!=0):
               if (l!=ind):
                   e=en
               else:
                   e=eb
               deltajmin=10000000
               deltajmax=0
               deltajmean=0
               for r in range(m-1):
                  delta[r,l]=(1-e*beta)*delta[r,l]+e*beta*(np.abs(x[r]-C[r,l]))
                  if (delta[r,l]<=deltajmin):
                      deltajmin=delta[r,l]
                  if(delta[r,l]>=deltajmax):
                      deltajmax=delta[r,l]
                  deltajmean=deltajmean+delta[r,l]/(m-1)
             
             
               if(deltajmin!=deltajmax):
                  for r in range(m-1):
                     W[r,l]=1.0/(1.0+np.exp((deltajmean-delta[r,l])/(s*(deltajmax-deltajmin))))
               else:
                  W[:,l]=1.0
            
               for r in range(m-1):
                  C[r,l]=C[r,l]+e*(x[r]-C[r,l])
         wins[ind]=wins[ind]+1 
         
        
         ### grafico ###
    
         fig0 = plt.figure(figsize=(10, 10))
         ax0 = plt.axes(projection="3d")
         ax0.scatter3D(D1[0,:],D1[1,:],D1[2,:], c='red', alpha=1)
         ax0.scatter3D(D2[0,:],D2[1,:],D2[2,:], c='green', alpha=1)
         ax0.scatter3D(D3[0,:],D3[1,:],D3[2,:], c='blue', alpha=1)
    
         # [view_init] Modifica o ângulo de visualização do gráfico
         #ax0.view_init(50, 35)
         #plt.show()
         i=0
         cont=0
         for r in range(Nmax):
            if(C[m-1,r]!=0):
               cont=cont+1
         Nos=np.zeros((m-1,cont))
         for r in range(Nmax):
            if(C[m-1,r]!=0):
               Nos[0,i]=C[0,r]
               Nos[1,i]=C[1,r]
               Nos[2,i]=C[2,r]
               i=i+1
             
         for r in range(Nmax):
            for s in range(Nmax):
               if((((C[m-1,r]!=0) and (C[m-1,s]!=0)) and (s!=r)) and (conexao[r,s]==1)):
                  xplot=[C[0,r],C[0,s]]
                  yplot=[C[1,r],C[1,s]]
                  zplot=[C[2,r],C[2,s]]
                  ax0.plot3D(xplot,yplot,zplot,c='black')
         X=Nos[0,:]
         Y=Nos[1,:]
         Z=Nos[2,:]
         ax0.scatter3D(X, Y, Z, c='black', s=70, alpha=1)
         ax0.set_title('SSSOM - Convergence phase', fontsize=18)
         ax0.set_xlabel('X', fontsize=15)
         ax0.set_ylabel('Y', fontsize=15)
         ax0.set_zlabel('Z', fontsize=15)# [view_init] Modifica o ângulo de visualização do gráfico
         ax0.view_init(50, 35)
         plt.show()
