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
cls1 = [0,1,2,3,4]
arvore = [[1,4],[2,3]]
tamanhos = np.zeros(qtc)
bd = []
avgMulticlassCFM = np.zeros((qtc,qtc))
avgTreeCFM = np.zeros((qtc,qtc))
for j,i in enumerate(cls1):
        ant = len(bd)
        bd += ler_arquivo("Base Multiclasse/classe_{}.txt".format(i))
        tamanhos[j] = len(bd) - ant
print tamanhos

labels = []
#print tamanhos
for k,i in enumerate(tamanhos):
        for j in range(int(i)):
                labels.append(k)
                
for it in range(50):
        TrainAtt = []
        TestAtt = []
        qtdTreino = 70
        qtdTeste = 30
        qtdtreinopc = np.zeros(qtc)
        qtdtestepc = np.zeros(qtc)
        while (len(TrainAtt)<(qtdTreino*qtc) + 210):
                rd = random.randint(0,len(bd)-1)
                if rd not in TrainAtt:
                        if qtdtreinopc[int(labels[rd])] < qtdTreino:
                                TrainAtt.append(rd)
                                qtdtreinopc[int(labels[rd])] +=1
                        elif labels[rd] == 0:
                                if qtdtreinopc[int(labels[rd])] < 280:
                                        TrainAtt.append(rd)
                                        qtdtreinopc[int(labels[rd])] +=1                        
        
        while (len(TestAtt)<qtdTeste*qtc):
                rd = random.randint(0,len(bd)-1)
                if rd not in TrainAtt and rd not in TestAtt:
                        if qtdtestepc[int(labels[rd])] < qtdTeste:
                                TestAtt.append(rd)
                                qtdtestepc[int(labels[rd])] += 1                        
        train = []
        trainLabel = []
        for i in TrainAtt:
                train.append(bd[i])
                trainLabel.append(labels[i])
        svmMulticlass = cv2.SVM()
        params = dict(kernel_type = cv2.SVM_RBF,
                      svm_type = cv2.SVM_C_SVC,
                          C=9.0,
                          gamma = 9.0,
                          term_crit = (cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-16))
        svmMulticlass.train_auto(np.float32(train),np.float32(trainLabel),None,None,params = params, k_fold=2)
        svmMulticlassCFM = np.zeros((qtc,qtc))
        svmTreeCFM = np.zeros((qtc,qtc))
        for i in TestAtt:
                esperado = int(labels[i])
                medido = int(svmMulticlass.predict(np.float32(bd[i])))
                svmMulticlassCFM[esperado,medido] += 1
        train = []
        trainLabel = []
        for i in TrainAtt:
                train.append(bd[i])
                if labels[i]>=1:
                        trainLabel.append(1)
                else: trainLabel.append(labels[i])
        svmRaiz = cv2.SVM()
        svmRaiz.train_auto(np.float32(train),np.float32(trainLabel),None,None,params = params, k_fold=2)
        train = []
        trainLabel = []
        for i in TrainAtt:
                if (labels[i] in arvore[0]):
                        train.append(bd[i])
                        trainLabel.append(0)
                elif (labels[i] in arvore[1]):
                        train.append(bd[i])
                        trainLabel.append(1)                
        svmNode1 = cv2.SVM()
        svmNode1.train_auto(np.float32(train),np.float32(trainLabel),None,None,params = params, k_fold=2)
        train = []
        trainLabel = []
        for i in TrainAtt:
                if (labels[i] == arvore[0][0]):
                        train.append(bd[i])
                        trainLabel.append(0)
                elif (labels[i] == arvore[0][1]):
                        train.append(bd[i])
                        trainLabel.append(1)  
        svmLeaf1 = cv2.SVM()
        svmLeaf1.train_auto(np.float32(train),np.float32(trainLabel),None,None,params = params, k_fold=2)
        train = []
        trainLabel = []
        for i in TrainAtt:
                if (labels[i] == arvore[1][0]):
                        train.append(bd[i])
                        trainLabel.append(0)
                elif (labels[i] == arvore[1][1]):
                        train.append(bd[i])
                        trainLabel.append(1)
        svmLeaf2 = cv2.SVM()
        svmLeaf2.train_auto(np.float32(train),np.float32(trainLabel),None,None,params = params, k_fold=2)
        
        for i in TestAtt:
                esperado = int(labels[i])
                medido = int(svmRaiz.predict(np.float32(bd[i])))
                if(medido ==1 ):
                        medido = int(svmNode1.predict(np.float32(bd[i])))
                        if medido == 0:
                                medido = int(svmLeaf1.predict(np.float32(bd[i])))
                                if medido == 0:
                                        medido = 1
                                elif medido == 1:
                                        medido = 4
                        elif medido == 1:
                                medido = int(svmLeaf2.predict(np.float32(bd[i])))
                                if medido == 0:
                                        medido = 2
                                elif medido == 1:
                                        medido = 3                        
                        else: print "error"
                
                svmTreeCFM[esperado,medido] += 1
        avgMulticlassCFM += svmMulticlassCFM
        avgTreeCFM += svmTreeCFM
avgMulticlassCFM =  np.divide(avgMulticlassCFM,it+1)
avgTreeCFM =  np.divide(avgTreeCFM,it+1)
acuracia = 0
for j,i in enumerate(avgMulticlassCFM):
        acuracia+= avgMulticlassCFM[j,j]
        i = np.divide(i,30)
        print i
print "ACERTO GERAL: ",acuracia*100/(qtc*30)
print;print
acuracia = 0
for j,i in enumerate(avgTreeCFM):
        acuracia+= avgTreeCFM[j,j]
        i = np.divide(i,30)
        print i
print "ACERTO GERAL: ",acuracia*100/(qtc*30)                