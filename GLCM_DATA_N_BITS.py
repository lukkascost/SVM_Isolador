from MachineLearn.Classes.Extractors.GLCM import *

for nbits in range(1,9):
        arrayGLCM = np.zeros((0,25))            
        for i in range(5):
                fname = "../isolador-multiclasse/base/tempo/filtrado_lucas/{}b/Classe_{}.txt.gz".format(nbits,i)
                
                print "Creating array of objects..."
                glArray = np.array([GLCM(np.matrix(x), nbits) for x in np.loadtxt(fname, delimiter=",")],dtype="object")
                
                print "generate co occurence matrix..."
                [gl.generateCoOccurenceHorizontal() for gl in glArray]
                
                print "Normalizing co occurence ..."
                [gl.normalizeCoOccurence() for gl in glArray]
                
                print "Calculating attributes..."
                [gl.calculateAttributes() for gl in glArray]
                
                print "Exporting attributes"
                for gl in glArray:
                        arrayGLCM = np.vstack((arrayGLCM,gl.exportToClassfier(float(i+1)))) 
                
                print "Finish!"
                
                print arrayGLCM
                print "\n"
        np.savetxt("GLCM_FILES/{}b.txt.gz".format(nbits),arrayGLCM,delimiter = ",", fmt="%.10e")
        print "File Generated successful"