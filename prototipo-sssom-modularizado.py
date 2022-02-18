
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

### variaveis globais (constantes) e outras variaveis ###

global n,m,S,at,eb,en,beta,c,Nmax,lp,tmax,push_rate,noclass,age_wins,epsilon,s

n=30 # quantidade de dados em cada cluster
m=4 # dimensao + 1 (rotulo)

S=n*3
at=0.9
eb=0.0005
en=0.002*eb
beta=0.01
c=0.5
#minwd=0.25 --- c?
Nmax=S
#maxcomp=2*S --- age_wins?
lp=0.01
tmax=10*S+50
N=0
push_rate=0.05*eb # ew
noclass=999
nwins=1
age_wins=S
epsilon=0.01
s= 0.05 ## no intervalo [0.01,0.1]

### dados artificiais ###

def geradados():
   D1=np.zeros((m,n))
   D2=np.zeros((m,n))
   D3=np.zeros((m,n))
   #rd.seed(1)

   D1[m-1,:]=1 #999
   D2[m-1,:]=2 #999
   D3[m-1,:]=3 #999

   rd.seed(11)

   for i in range(n):
       D1[0,i]=rd.normalvariate(1.0,0.15)
       D2[0,i]=rd.normalvariate(0.5,0.15)
       D3[0,i]=0.0 #rd.normalvariate(5,0.25)
       D1[1,i]=rd.normalvariate(0.5,0.15)
       D2[1,i]=0.0 #rd.normalvariate(0,0.25)
       D3[1,i]=rd.normalvariate(1.0,0.15)
       D1[2,i]=0.0#rd.normalvariate(5,0.25)
       D2[2,i]=rd.normalvariate(1.0,0.15)
       D3[2,i]=rd.normalvariate(0.5,0.15)


   #D1[m-1,n-int(n/3):n+1]=1
   #D2[m-1,n-int(n/3):n+1]=2
   #D3[m-1,n-int(n/3):n+1]=3
   
   return D1,D2,D3
   
### grafico dados artificiais ###
 
def plotadados(D1,D2,D3): 
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

def sorteiodados(D1,D2,D3):
   k=rd.randint(0,3*n-1)
   if((k>=0) and (k<n)):
       x=D1[:,k]
   elif((k>=n) and (k<2*n)):
       x=D2[:,k-n]
   else:
       x=D3[:,k-2*n]
   return x

def ativacao_s1(cont,W,x,C,a_s,ind,j,s1,N):
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
         #########################
      elif((j==0) and (C[m-1,i]==0)):
         j=i
   ### winner ###    
   s1=C[:,ind]
   N=cont
   return a_s,s1,N,j,ind

def conectar_todos(W,C,classj,conexao):
   for l in range(Nmax):
      for r in range(Nmax):
         normaw=np.linalg.norm(W[:,r]-W[:,l])
         classr=C[m-1,r]
         if (((((classr==classj) or (classr==noclass)) or(classj==noclass)) and (normaw<c*(m**0.5))) and (classr!=0)):
            conexao[r,l]=1
         else:
            conexao[r,l]=0
   return conexao

def conectar_um(W,C,classj,conexao,ind):
   for r in range(Nmax):
      normaw=np.linalg.norm(W[:,r]-W[:,ind])
      classr=C[m-1,r]
      if (((((classr==classj) or (classr==noclass)) or(classj==noclass)) and (normaw<c*(m**0.5))) and (classr!=0)):
         conexao[r,ind]=1
      else:
         conexao[r,ind]=0
   return conexao

def atualizar_nos(ind,delta,x,C,W):
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
   return delta,W,C

def atualizar_nos_s2(delta,W,C,ind, ind2,xpush_rate):
   delta,W,C =atualizar_nos(ind2,delta,x,C,W)
   
   for l in range(Nmax):
      if((l==ind) and (conexao[l,ind2]==0)):
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
   return delta,W,C

def achar_s2(C,x,cont,W,a_s,j,s1,s2):
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
         #########################
      elif((j==0) and (C[m-1,z]==0)):
         j=z
   N=cont
   if ((C[:,ind2]!=s1) and (a_s>=at)):
      s2=C[:,ind2]
   return N,s2,ind2,j,a_s

def remover_nos(wins,conexao,C):
   for r in range(Nmax):         
      if(wins[r]<lp*age_wins):
         conexao[r,:]=0
         conexao[:,r]=0
         C[:,r]=0
      wins[r]=0
   return conexao,C,wins

def modo_supervisionado(delta,s1,a_s,N,C,W,wins,x,conexao,ind,j):
   if((s1[m-1]==noclass)or(s1[m-1]!=0)):
      if ((a_s<at) and (N<Nmax)):
         C[:,j]=x[:]
         W[:,j]=1.0
         wins[j]=0
         classj=x[m-1]            
         ### update connections ###
         conexao=conectar_todos(W,C,classj,conexao)
         ##########################            
      elif(a_s>=at):             
         #### update nodes ###
         delta,W,C = atualizar_nos(ind,delta,x,C,W)
         ####################            
         s1[m-1]=x[m-1]  
         classj=s1[m-1]            
         #### update conections with one node ####
         conexao = conectar_um(W,C,classj,conexao,ind)
         #########################################            
         wins[ind]=wins[ind]+1        
   else:   
      s2=np.zeros(m)
      cont=0
      j=0
      ##### new winner ####
      N,s2,ind2,j,a_s = achar_s2(C,x,cont,W,a_s,j,s1,s2)
      #####################           
      if (s2[m-1]!=0):
         ### update s2 ###
         delta,W,C = atualizar_nos_s2(delta,W,C,ind, ind2,x)
         ###################
         wins[ind2]=wins[ind2]+1       
      elif(N<Nmax):
         C[:,j]=x[:]
         W[:,j]=1.0
         wins[j]=0
         classj=x[m-1]
         ### connect j to other nodes ###
         conexao=conectar_um(W,C,classj,conexao,j)
         ################################
   return N,delta,C,W,wins,conexao

def modo_nao_supervisionado(C,N,a_s,W,wins,x,conexao,delta,ind,beta,j):
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
      ### connect j to the other nodes ###
      conexao=conectar_um(W,C,classj,conexao,j)
      #####################################
   else:
      #### update node ###
      delta,W,C=atualizar_nos(ind,delta,x,C,W)
      ####################
      wins[ind]=wins[ind]+1
   return N,delta,C,W,wins,conexao

def graficos(D1,D2,D3,C,titulo):
   fig0 = plt.figure(figsize=(10, 10))
   ax0 = plt.axes(projection="3d")
   ax0.scatter3D(D1[0,:],D1[1,:],D1[2,:], c='red', alpha=1)
   ax0.scatter3D(D2[0,:],D2[1,:],D2[2,:], c='green', alpha=1)
   ax0.scatter3D(D3[0,:],D3[1,:],D3[2,:], c='blue', alpha=1)
   ax0.set_title('Dados', fontsize=18)
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
   ax0.set_title(titulo, fontsize=18)
  
   ax0.set_xlabel('X', fontsize=15)
   ax0.set_ylabel('Y', fontsize=15)
   ax0.set_zlabel('Z', fontsize=15)# [view_init] Modifica o ângulo de visualização do gráfico
   ax0.view_init(50, 35)
   plt.show()


def remover_nos_fase_de_convergencia(wins,conexao,C,remocoes):
   for r in range(Nmax):         
      if(wins[r]<lp*age_wins):
         remocoes=1
         conexao[r,:]=0
         conexao[:,r]=0
         C[:,r]=0
      else:
         print("nao removeu",wins[r])
   return C,conexao,remocoes

def fase_de_convergencia(C,wins,W,conexao,x,a_s,s1,ind,j,N,delta,beta,D1,D2,D3):
   remocoes=1  
   nmax=0 
   aux=0 
   while (remocoes!=0):
      cont=0
      remocoes=0
      for i in range(Nmax):
         if(C[m-1,i]!=0):
             cont=cont+1
      nmax=cont
      if(nmax>1):
         ### remove nodes ###            
         C,conexao,remocoes=remover_nos_fase_de_convergencia(wins,conexao,C,remocoes)
         ####################
      cont=0
      for i in range(Nmax):
         if(C[m-1,i]!=0):
             cont=cont+1
      N=cont
      if((N==nmax) or (N==1)):
         remocoes=0
         classj=x[m-1]
      else:
         classj=noclass
      print('num de remocoes= ',remocoes)
      print('wins= ',wins)
      if(remocoes!=0):
         ### update the connections with all nodes ###
         conexao=conectar_todos(W,C,classj,conexao)
         ##############################################
         wins[:]=0
         for t in range(tmax):
            ### random input ###
            x=sorteiodados(D1,D2,D3)
            ####################
            ### compute the activation of all nodes ###
            a_s,s1,aux,j,ind=ativacao_s1(cont,W,x,C,a_s,ind,j,s1,N)    
            ###########################################            
            #### update node ###
            delta,W,C=atualizar_nos(ind,delta,x,C,W)
            ####################
            ### grafico ###
            graficos(D1,D2,D3,C,'SSSOM - Convergence phase')
            ###############
   return W

def iniciar(C,x,N):
   C[:,0]=x[:]
   N=1
   if((x[m-1]!=noclass) and (x[m-1]!=0)):
       classj=x[m-1]
   elif(x[m-1]!=0):
       classj=noclass
   return C,N,classj 

def global_relevance(W,C,w):
   for i in range(m-1):
      maximo=0
      variancia=0
      media=0
      for j in range(Nmax):
         media=media+(C[i,j]/Nmax)
         if(W[i,j]>=maximo):
            maximo=W[i,j]
      for j in range(Nmax):
         variancia=variancia+(((C[i,j]-media)**2.0)/Nmax)
      w[i]=maximo*variancia    
   return w


### dados ###

D1,D2,D3=geradados()
plotadados(D1,D2,D3)

### self-organization phase ###

C=np.zeros((m,Nmax))
delta=np.zeros((m-1,Nmax))
W=np.zeros((m-1,Nmax))
wins=np.zeros(Nmax)
conexao=np.eye(Nmax)
w=np.zeros(m-1)

x=sorteiodados(D1,D2,D3)
   
C,N,classj = iniciar(C,x,N)

for t in range(tmax+1):
   ### random input ###
   x=sorteiodados(D1,D2,D3)
   ####################
   a_s=0
   cont=0
   j=0
   ind=0
   s1=np.zeros(m)
   a_s,s1,N,j,ind = ativacao_s1(cont,W,x,C,a_s,ind,j,s1,N)   
   if((x[m-1]!=noclass) and (x[m-1]!=0)):
      #### supervised mode ####
      N,delta,C,W,wins,conexao = modo_supervisionado(delta,s1,a_s,N,C,W,wins,x,conexao,ind,j)
      #########################
   else:
      #### unsupervised mode ####
      N,delta,C,W,wins,conexao = modo_nao_supervisionado(C,N,a_s,W,wins,x,conexao,delta,ind,beta,j)
      ###########################  
   if(nwins==age_wins):
      ### remove nodes ###            
      conexao,C,wins=remover_nos(wins,conexao,C)
      ####################
      nwins=0
   nwins=nwins+1
   ### grafico ###
   graficos(D1,D2,D3,C,'SSSOM')
   ###############
         
#### Convergence phase ####
W=fase_de_convergencia(C,wins,W,conexao,x,a_s,s1,ind,j,N,delta,beta,D1,D2,D3)
###########################

#### global relevance ####
w=global_relevance(W,C,w)
##########################

print('w=',w)
print('W=',W)
print('C=',C)