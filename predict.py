import cv2
import numpy as np
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
        return bd                                                       ##

Teste = ler_arquivo("dataTest.dat")
TesteLabel = []
Treino = ler_arquivo("dataTrain.dat")
TreinoLabel = []
for i,j in enumerate(Teste):
        TesteLabel.append(Teste[i][-1])
        del Teste[i][-1]
for i,j in enumerate(Treino):
        TreinoLabel.append(Treino[i][-1])
        del Treino[i][-1]

#cv2.TERM_CRITERIA_MAX_ITER = 10

svm_params = dict(kernel_type = cv2.SVM_LINEAR,
                  svm_type = cv2.SVM_C_SVC,
                  C=9,
                  term_crit = (cv2.TERM_CRITERIA_MAX_ITER, 1,  1.1920928955078125e-07)
                  
                  )
svm = cv2.SVM()
svm.train(np.float32(Treino),np.float32(TreinoLabel),params = svm_params)
acerto = 0
total= 0
svm.save("Vetores_svm.txt")
for i,j in enumerate(Teste):
        total+=1
        resultado_medido = svm.predict(np.float32(j))
        resultado_real = TesteLabel[i]
        if(resultado_medido==resultado_real):
                acerto+=1
print acerto/total