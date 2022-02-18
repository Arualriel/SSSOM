import numpy as np


### dados reais ###


lista_de_dados=['data-prototipo-sssom/breast.arff','data-prototipo-sssom/diabetes.arff','data-prototipo-sssom/glass.arff','data-prototipo-sssom/pendigits.arff','data-prototipo-sssom/shape.arff','data-prototipo-sssom/vowel.arff','data-prototipo-sssom/liver.arff']
dados=lista_de_dados[2]
with open(dados,'r') as arquivo:
    linhas = arquivo.read().splitlines()
with open(dados,'r') as arquivo:
    linha = arquivo.read()
m=int(linhas[0])+1 # dimensao + rotulo

S=len(linhas)-1 # quantidade de dados
Dado=np.zeros((m,S))

i=0
j=0
num=''
aux=0
for linha in linhas:
    if aux!=0:
        i=0
        for l in linha:
            if (l == ','):
                Dado[i,j]=float(num)
                l=''
                num=''
                i=i+1
            elif(i<m-1):
                print(i,m)
                num=num+l
            elif(i==m-1):
                print(i,m)
                Dado[i,j]=float(l)
                i=i+1
        j=j+1
    aux=1
     
print(Dado)
