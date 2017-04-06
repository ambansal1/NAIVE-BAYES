import os
import csv
import numpy as np
import NB

# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('..','data')

# Read vocabulary into a list
# You will not need the vocabulary for any of the homework questions.
# It is provided for your reference.
with open(os.path.join(data_dir, 'vocabulary.csv'), 'rb') as f:
    reader = csv.reader(f)
    vocabulary = list(x[0] for x in reader)

# Load numeric data files into numpy arrays
XTrain = np.genfromtxt(os.path.join(data_dir, 'XTrain.csv'), delimiter=',')
yTrain = np.genfromtxt(os.path.join(data_dir, 'yTrain.csv'), delimiter=',')
XTrainSmall = np.genfromtxt(os.path.join(data_dir, 'XTrainSmall.csv'), delimiter=',')
yTrainSmall = np.genfromtxt(os.path.join(data_dir, 'yTrainSmall.csv'), delimiter=',')
XTest = np.genfromtxt(os.path.join(data_dir, 'XTest.csv'), delimiter=',')
yTest = np.genfromtxt(os.path.join(data_dir, 'yTest.csv'), delimiter=',')

D = []
alpha = 5
beta = 2
#yHat = [1,0,0,0,0,0,0]
#yTruth = [1,0,0,0,1,1]

# TODO: Test logProd function, defined in NB.py
#x = [1,2,3,4,5]
#uy = NB.logProd(x)
#print uy

# TODO: Test NB_XGivenY function, defined in NB.py
D = NB.NB_XGivenY(XTrainSmall, yTrainSmall, alpha, beta)
print D
E = np.where( D > 0.3 )[0]
F = np.where( D < 0.1 )[0]
print "lengthof E"
print len(E) + len(F)
# TODO: Test NB_YPrior function, defined in NB.py
p = NB.NB_YPrior(yTrainSmall)
print p
#D = np.zeros([2, XTrain.shape[1]])
#p =0.7
# TODO: Test NB_Classify function, defined in NB.py
yHat = NB.NB_Classify(D,p,XTest)
print "yHat = "
print yHat
# TODO: Test classificationError function, defined in NB.py
#yTestt = yTest
#print yTest
#yTestt[0:100] = 0
#print yTestt
error = NB.classificationError(yHat,yTest)
print "error ="
print error
# TODO: Run experiments outlined in HW2 PDF