from MachineLearn.Classes.Extractors.GLCM import *

fname = "../isolador-multiclasse/base/tempo/filtrado_lucas/Classe_0.txt.gz"
nbits = 12

print "Creating array of objects..."
glArray = np.array([GLCM(np.matrix(x), nbits) for x in np.loadtxt(fname, delimiter=",")],dtype="object")

print "generate co occurence matrix..."
[gl.generateCoOccurenceHorizontal() for gl in glArray]

print "Normalizing co occurence ..."
[gl.normalizeCoOccurence() for gl in glArray]

print "Calculating attributes..."
[gl.calculateAttributes() for gl in glArray]

print "Finish!"