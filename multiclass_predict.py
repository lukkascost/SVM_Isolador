import cv2
import numpy as np
from Classes import *
def ler_arquivo(address):
        arquivo = open(address,"r")                                     ##
        bd = []                                                         ##
        obj = 0                                                         ##
        for line in arquivo:                                            ##
                bd.append([])                                           ##
                lines = line.split(",")                                 ##
                for attr in lines:                                      ##
                        if(len(attr)>=1): bd[obj].append(float(attr))    ##
                obj = obj+1                                             ##
        return bd  
tamanhos = np.zeros(5)
Teste = ler_arquivo("Base Multiclasse/classe_0.txt")
tamanhos[0] = len(Teste)
Teste += ler_arquivo("Base Multiclasse/classe_1.txt")
tamanhos[1] = len(Teste) - sum(tamanhos[:1])
Teste += ler_arquivo("Base Multiclasse/classe_2.txt")
tamanhos[2] = len(Teste) - sum(tamanhos[:2])
Teste += ler_arquivo("Base Multiclasse/classe_3.txt")
tamanhos[3] = len(Teste) - sum(tamanhos[:3])
Teste += ler_arquivo("Base Multiclasse/classe_4.txt")
tamanhos[4] = len(Teste) - sum(tamanhos[:4])
labels = []
print tamanhos
for k,i in enumerate(tamanhos):
        for j in range(int(i)):
                labels.append(k)
oRodada = rodada(50, 5, nTreino=50, nAtrib=12, amostra=200)

for i,j in enumerate(Teste):
        oRodada.GLCM.add_objeto(Teste[i]+[labels[i]+1])
params = dict(kernel_type = cv2.SVM_LINEAR,
                  svm_type = cv2.SVM_C_SVC,
                  C=9.0,
                  gamma = 9.0,
                  term_crit = (cv2.TERM_CRITERIA_MAX_ITER, 1, 1e-16))

#oRodada.execIteractions([x for x in range(12)],params=svm_params)
self = oRodada
typeRandom = 1
self.clean()
positions = [x for x in range(12)]
for i in range(self.num_ite):
        self.iteracoes[i].conj_treino,self.iteracoes[i].conj_teste = self.GLCM.extraiTp1(150, 50, 5)
        train = []
        trainLabel =[]
        for k in self.iteracoes[i].conj_treino:
                train.append(self.GLCM.getNewAtrib(k,positions))
                trainLabel.append(self.GLCM.labels[k])
        
        if params is None:
                self.iteracoes[i].svm_params = dict(kernel_type = cv2.SVM_RBF,svm_type = cv2.SVM_C_SVC,gamma=2.0)
        else: self.iteracoes[i].svm_params = params
        svm = cv2.SVM()
        svm.train(np.float32(train),np.float32(trainLabel),params = self.iteracoes[i].svm_params)
        for k in self.iteracoes[i].conj_teste:
                sample = np.float32(self.GLCM.getNewAtrib(k, positions))
                res = int(svm.predict(sample))
                test = int(self.GLCM.labels[k])
                self.iteracoes[i].dados[test,res]+=1
        self.iteracoes[i].set_acuracia()
Salvar_texto(str(self),"resultados_multiclasse_50teste_200treino.txt")