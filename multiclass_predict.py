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
                        if(len(attr)>=1): bd[obj].append(float(attr))   ##
                obj = obj+1                                             ##
        return bd  
qtc = 5
maior = 0.80
clss = [0,1,2,3,4]
tamanhos = np.zeros(qtc)
Teste = []
for j,i in enumerate(clss):
        ant = len(Teste)
        Teste += ler_arquivo("Base Multiclasse/classe_{}.txt".format(i))
        tamanhos[j] = len(Teste) - ant
print tamanhos
labels = []
#print tamanhos
for k,i in enumerate(tamanhos):
        for j in range(int(i)):
                labels.append(k)
oRodada = rodada(500, qtc, nTreino=50, nAtrib=12, amostra=200)

for i,j in enumerate(Teste):
        oRodada.GLCM.add_objeto(Teste[i]+[labels[i]+1])
params = dict(kernel_type = cv2.SVM_RBF,
                  svm_type = cv2.SVM_C_SVC,
                  C=9.0,
                  gamma = 9.0,
                  term_crit = (cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-16))

#oRodada.execIteractions([x for x in range(12)],params=svm_params)
self = oRodada
typeRandom = 1
self.clean()
positions = [x for x in range(1,13)]
for i in range(self.num_ite):
        print i
        self.iteracoes[i].conj_treino,self.iteracoes[i].conj_teste = self.GLCM.extraiTp1(150, 50, qtc)
        train = []
        trainLabel =[]
        for k in self.iteracoes[i].conj_treino:
                train.append(self.GLCM.getNewAtrib(k,positions))
                trainLabel.append(self.GLCM.labels[k])
        if params is None:
                self.iteracoes[i].svm_params = dict(kernel_type = cv2.SVM_RBF,svm_type = cv2.SVM_C_SVC,gamma=2.0)
        else: self.iteracoes[i].svm_params = params
        svm = cv2.SVM()
        svm.train_auto(np.float32(train),np.float32(trainLabel),None,None,params = self.iteracoes[i].svm_params, k_fold=2)
        for k in self.iteracoes[i].conj_teste:
                sample = np.float32(self.GLCM.getNewAtrib(k, positions))
                res = int(svm.predict(sample))
                test = int(self.GLCM.labels[k])
                self.iteracoes[i].dados[test,res]+=1
        self.iteracoes[i].set_acuracia()
        self.sum_cfm = np.add(self.sum_cfm,self.iteracoes[i].dados) 
        if self.iteracoes[i].acuracia[2][0]> maior:
                svm.save("Vetores_svm2_3.txt")
                maior = self.iteracoes[i].acuracia[2][0]
                print "salvou",maior
Salvar_texto(str(self),"resultados_multiclasse_4_50teste_200treino.txt")

for i in self.get_avg_cfm():
        string = ""
        for j in i:
                string += "\t{:02.02f}".format(j)
        print string
                
