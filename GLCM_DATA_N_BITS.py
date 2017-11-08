from MachineLearn.Classes.Extractors.GLCM import *
from MachineLearn.Classes.data import Data
from MachineLearn.Classes.data_set import DataSet
from MachineLearn.Classes.experiment import Experiment


####################################################################################################################################
## TO GENERATE GLCM FILES

#for nbits in [8]:
        #arrayGLCM = np.zeros((0,25))            
        #for i in [0,1,2,3,4]:
                #fname = "../isolador-multiclasse/base/tempo/filtrado_lucas/{}b/Classe_{}.txt.gz".format(nbits,i)
                #array = np.loadtxt(fname, delimiter=",")
                #array[139,1477] = 255
                #print array[array>=256]
                
                #print "Creating array of objects..."
                #glArray = np.array([GLCM(np.matrix(x), nbits) for x in array],dtype="object")
                
                #print "generate co occurence matrix..."
                #[gl.generateCoOccurenceHorizontal() for gl in glArray]
                
                #print "Normalizing co occurence ..."
                #[gl.normalizeCoOccurence() for gl in glArray]
                
                #print "Calculating attributes..."
                #[gl.calculateAttributes() for gl in glArray]
                
                #print "Exporting attributes"
                #for gl in glArray:
                        #arrayGLCM = np.vstack((arrayGLCM,gl.exportToClassfier(float(i+1)))) 
                
                #print "Finish!"
                
                #print arrayGLCM
                #print "\n"
        #np.savetxt("GLCM_FILES/{}b.txt.gz".format(nbits),arrayGLCM,delimiter = ",", fmt="%.10e")
        #print "File Generated successful"
        
####################################################################################################################################
        
exp = Experiment()
niterations = 50



for nbits in [2,3,4,5,6,7,8]:
        fname = "GLCM_FILES/{}b.txt.gz".format(nbits)
        array = np.loadtxt(fname, delimiter=",")
        obDataSet = DataSet()
        for j in range(1,6):
                ar = array[array[:,-1]==j]
                np.random.shuffle(ar)
                ar = ar[:200]
                for i in ar:
                        obDataSet.addSampleOfAtt(i)
        for itIndex in range(niterations):
                obData = Data(5, 50, samples=200)
                obData.randomTrainingTest()
                svm = cv2.SVM()
                obData.params = dict(kernel_type = cv2.SVM_RBF,svm_type = cv2.SVM_C_SVC,gamma=2.0,nu = 0.0,p = 0.0, coef0 = 0)
                svm.train_auto(np.float32(obDataSet.atributes[obData.Training_indexes]),np.float32(obDataSet.labels[obData.Training_indexes]),None,None,obData.params,k_fold = 2)
                results =  svm.predict_all(np.float32(obDataSet.atributes[obData.Testing_indexes]),np.float32(obDataSet.labels[obData.Testing_indexes]))
                obData.setResultsForClassfier(results, obDataSet.labels[obData.Testing_indexes])        
                obDataSet.append(obData)
        exp.addDataSet(obDataSet, description="Test for {}bits database: ".format(nbits))
        print exp
        exp.save("EXPERIMENTS/EXP01-GLCM+SVM-2_8b.txt")
