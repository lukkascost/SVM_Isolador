import cv2
import numpy as np
from Classes import *

svm = cv2.SVM()
svm.load("RBF_Vetores/Vetores_svm01234.txt")
svm1 = cv2.SVM()
svm1.load("RBF_Vetores/Vetores_svm0_1234.txt")
svm2 = cv2.SVM()
svm2.load("RBF_Vetores/Vetores_svm14_23.txt")
svm3 = cv2.SVM()
svm3.load("RBF_Vetores/Vetores_svm1_4.txt")
svm4 = cv2.SVM()
svm4.load("RBF_Vetores/Vetores_svm2_3.txt")

qtc = 5
clss = [0,1,2,3,4]
resMulti = np.zeros((qtc,qtc))
resTree  = np.zeros((qtc,qtc))
tamanhos = np.zeros(qtc)
Teste = []
for j,i in enumerate(clss):
        ant = len(Teste)
        Teste += ler_arquivo("Base Multiclasse/classe_{}.txt".format(i))
        tamanhos[j] = len(Teste) - ant
labels = []
for k,i in enumerate(tamanhos):
        for j in range(int(i)):
                labels.append(k)
for i,j in enumerate(Teste):
        esperado = int(labels[i])
        medido   = svm1.predict(np.float32(j))
        ## define se é classe 0 ou 1 
        if medido == 1:
                medido = svm2.predict(np.float32(j))
                ## define se é 1,4 ou 2,3
                if medido == 0:
                        medido = svm3.predict(np.float32(j))
                        if medido == 0:resTree[esperado,1] += 1  
                        else: resTree[esperado,4] += 1
                elif medido == 1:
                        medido = svm4.predict(np.float32(j))
                        if medido == 0:resTree[esperado,2] += 1  
                        else: resTree[esperado,3] += 1                        
        else:
                resTree[esperado,0] += 1
        resMulti[esperado,int(svm.predict(np.float32(j)))] += 1

for i in resMulti:
        print i , "qtd = ", sum(i)
        
for i in resTree:
        print i , "qtd = ", sum(i)    
    
print "SVM Multiclasse"
geral= 0
for i in range(5):
        print "Acuracia classe {:02d}: {:03.05f}%".format(i,resMulti[i,i]*100/tamanhos[i])
        geral+=resMulti[i,i]
print "Acuracia Geral: {:03.05f}%".format(geral*100/sum(tamanhos))

print "SVM Arvore"
geral= 0
for i in range(5):
        print "Acuracia classe {:02d}: {:03.05f}%".format(i,resTree[i,i]*100/tamanhos[i])
        geral+=resTree[i,i]
print "Acuracia Geral: {:03.05f}%".format(geral*100/sum(tamanhos))