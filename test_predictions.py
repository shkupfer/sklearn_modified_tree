import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

X = np.asarray([(10.0,1.0,2.0,5.0,6.0),(100.0,2.0,3.0,3.0,2.0),(10.0,3.0,2.0,10.0,100.0), (50.0,1.0,3.0,1.0,3.0)])
X = X.astype(np.float32, copy=False)
Y = np.asarray([0,0,1,1])

X[:,0] = LabelEncoder().fit_transform(X[:,0])

print X

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)
predictions = regressor.predict(X)
print "BG output1: ", predictions
print
print
predictions_new = regressor.predict_new(X)
print "BG output2: ", predictions_new
