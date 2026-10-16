import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from PIL import Image

######################### SSIM CALCULADO NA IMAGEM E MSE CALCULADO NA MATRIZ! #########################################


n_autovetores = 7
grupo =  ['genero','expressao']
metodo = ['1_h','h','js','t_st','ssvm','pca']
m_metodo = len(metodo)
notacao_metodo = ['$\widehat{H}$','$H$','$JS$','$T$','$W$','$PCA$']
lista_imgs = np.arange(1,41)
npcaf = '359'
for g in grupo:
    
    comp_p = ['5','10','20','40','80','160','179','250','320',npcaf]
    cp = len(comp_p)
    for im in lista_imgs:
        # entre a imagem im original e imagem im reconstruida por todos os métodos
        matriz_mse = np.zeros((cp,m_metodo))
        matriz_ssim = np.zeros((cp,m_metodo))
        
        img1 = Image.open('visualizacao/corrigido/geral-FERET/original/imagem_original-' + g + '_n' + str(im) + '.png') 
        I1 = np.asarray(img1)
        max1 = np.amax(I1)
        min1 = np.amin(I1)
        
        img1_mse =[]
                    
        arq1 = open('visualizacao/corrigido/geral-FERET/original/imagem_original-' + g + '_n' + str(im) + '.txt')
        linhas1 = arq1.readlines()
        for linha in linhas1:
            lista_aux = []
            linha = linha.split(',')
            for item in linha:
            	img1_mse.append(float(item))
        arq1.close()  
        #print(img1_mse)    
        I1_mse = np.zeros((128,128))
        i = 0     
        k = 0
        for j in range(16384):
            if (k < 128):
           	    I1_mse[k,i] = img1_mse[j]
           	    k = k + 1
            elif(i < 128):
           	    k = 0
           	    i = i + 1
           	    I1_mse[k,i] = img1_mse[j]
           	    k = k + 1
        #print(I1_mse)
        #fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6,4),
        #                 sharex=True, sharey=True)
        #ax1.imshow(I1_mse, cmap = plt.cm.gray)
        #plt.show()       
        
        for c in range(cp):
            # carregando a reconstrução para a imagem com c componentes principais
            for m in range(m_metodo):
                if((metodo[m] == '1_h') or (metodo[m] == 'h')):
                    img2 = Image.open('visualizacao/corrigido/geral-FERET/reconstrucao/imagem_reconstruida_' + metodo[m] + '-' + g + '_npca' + comp_p[c] + '_n' + str(im) + '.png')
                else:
                    img2 = Image.open('visualizacao/corrigido/geral-FERET/reconstrucao/imagem_reconstruida_' + metodo[m] + '_npca' + comp_p[c] + '-' + g + '_n' + str(im) + '.png')
                #img2 = Image.open('visualizacao/corrigido/geral-FERET/reconstrucao/imagem_reconstruida_n' + str(im) + metodo[m] + '-' + g + '_npca' + comp_p[c] + '.png') 
                I2 = np.asarray(img2)
                min2 = np.amin(I2)
                max2 = np.amax(I2)
                
                img2_mse =[]
                    
                if((metodo[m] == '1_h') or (metodo[m] == 'h')):
                    arq2 = open('visualizacao/corrigido/geral-FERET/reconstrucao/imagem_reconstruida_' + metodo[m] + '-' + g + '_npca' + comp_p[c] + '_n' + str(im) + '.txt')
                else:
                    arq2 = open('visualizacao/corrigido/geral-FERET/reconstrucao/imagem_reconstruida_' + metodo[m] + '_npca' + comp_p[c] + '-' + g + '_n' + str(im) + '.txt')
                #arq2 = open('visualizacao/corrigido/geral-FERET/reconstrucao/imagem_reconstruida_n' + str(im) + '_' + metodo[m] + '-' + g + '_npca' + comp_p[c] + '.txt')
                linhas2 = arq2.readlines()
                for linha in linhas2:
                    lista_aux = []
                    linha = linha.split(',')
                    for item in linha:
                        lista_aux.append(float(item))
                    img2_mse.append(lista_aux)
                arq2.close()  
				           
                I2_mse = np.zeros((128,128))
                for i in range(128):
                    I2_mse[i,:] = img2_mse[i]
                   	
                #print(I2_mse)
                matriz_mse[c,m] = mean_squared_error(I1_mse,I2_mse)
                
                matriz_ssim[c,m] = ssim(I1,I2,win_size = 3,data_range = max2 - min2)
                
                fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6,4),
                         sharex=True, sharey=True)
                ax1.imshow(I1, cmap = plt.cm.gray)
                ax1.set_xlabel(f'MSE:{matriz_mse[c,m]:.3f}, SSIM:{matriz_ssim[c,m]:.3f}')
                ax1.set_title('Imagem reconstruída pelo ' + notacao_metodo[m] + ' com ' + comp_p[c] + ' componentes principais')
       			
                plt.tight_layout()
                #plt.show()
                fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(6,4),
                         sharex=True, sharey=True)
                ax2.imshow(I2, cmap = plt.cm.gray)
                ax2.set_xlabel(f'MSE:{matriz_mse[c,m]:.3f}, SSIM:{matriz_ssim[c,m]:.3f}')
                ax2.set_title('Imagem reconstruída pelo ' + notacao_metodo[m] + ' com ' + comp_p[c] + ' componentes principais')
       			
                plt.tight_layout()
                plt.savefig('visualizacao/corrigido/geral-FERET/mse_ssim_rec_todos_x_original/visualizacao_mse_ssim_original_x_' + metodo[m] + '-' + g + '_n' + str(im) + '_npca' + comp_p[c] + '.png', format='png')
                
                #plt.show()
                
        np.savetxt('visualizacao/corrigido/geral-FERET/mse_ssim_rec_todos_x_original/visualizacao_mse_original_x_metodos-' + g + '_n' + str(im) + '.txt',matriz_mse)
        np.savetxt('visualizacao/corrigido/geral-FERET/mse_ssim_rec_todos_x_original/visualizacao_ssim_original_x_metodos-' + g + '_n' + str(im) + '.txt',matriz_ssim)
                    


