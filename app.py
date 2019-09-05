import sys
import pandas as pd
sys.path.insert(0, "function/")


from DataExtractor import *
from RegClf import *


#data extraction
data = extract()
X_train, X_test, Y_train, Y_test=data.StockHistory()

#fit and choose are regressions
regression= RegClf(X_train, X_test, Y_train, Y_test)

#choose are regression types

regression.clftype([2,3,4,5,6])
regression.fit()
regression.predict()
regression.score()
bestReg=regression.mostAccurate()

print(bestReg)

# clfreg = LinearRegression(n_jobs=-1)
# clfreg.fit(X_train, Y_train)
# dfreg = data.getRawExtract()
# y_pred = clfreg.predict(X_test)
#
#
#
#
# plotGraph(data.getRawExtract(),X_test,y_pred)
#
#
# # Quadratic Regression 2
# clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
# clfpoly2.fit(X_train, Y_train)
#
# # Quadratic Regression 3
# clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
# clfpoly3.fit(X_train, Y_train)
#
# #
# #
# confidencereg = clfreg.score(X_test, Y_test)
# confidencepoly2 = clfpoly2.score(X_test, Y_test)
# confidencepoly3 = clfpoly3.score(X_test,Y_test)
#
# print(confidencereg)
# print(confidencepoly2)
# print(confidencepoly3)