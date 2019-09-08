import sys
import pandas as pd
sys.path.insert(0, "function/")


from DataExtractor import *
from RegClf import *


#data extraction
data = DataExtractor()
X_train, X_test, Y_train, Y_test=data.extract()

#fit and choose are regressions
regression= RegressionClassifier(X_train, X_test, Y_train, Y_test)

#choose are regression types

regression.defineClassifierType([2, 3, 4, 5, 6])
regression.fit()
regression.predict()
regression.score()
bestReg=regression.mostAccurateScore()
regression.plotGraph(data.getRawExtract())