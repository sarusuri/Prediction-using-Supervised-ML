import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


dataset=pd.read_csv('/Users/satishsuri/Downloads/studentScores.csv')

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

print(regressor.intercept_)

print(regressor.coef_)

y_pred_perc=(y_pred/100)*100
y_pred_perc=regressor.predict(X_test)
df=pd.DataFrame({'Actual':Y_test,'Predicted Percentage':np.round_(y_pred_perc,decimals=2,out=None)})
df

new_input=[[9.25]]
new_output=regressor.predict(new_input)
print(new_output)
