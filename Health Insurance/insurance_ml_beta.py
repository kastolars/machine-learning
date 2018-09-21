import pandas as pd
import numpy as np
 
data = pd.read_csv('insurance.csv')

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

male = data.loc[data['sex'] == 'male']
male = male.drop(columns=['sex'])
female = data.loc[data['sex'] == 'female']
female = female.drop(columns=['sex'])

maleSmoker = male.loc[male['smoker'] == 'yes']
maleSmoker = maleSmoker.drop(columns=['smoker'])
maleNotSmoker = male.loc[male['smoker'] == 'no']
maleNotSmoker = maleNotSmoker.drop(columns=['smoker'])
 
maleSmokerX = maleSmoker.iloc[:, 0].values
maleSmokerY = maleSmoker.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
maleSmokerX = sc_X.fit_transform(maleSmokerX.reshape(-1, 1))
maleSmokerY = sc_y.fit_transform(maleSmokerY.reshape(-1, 1))

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(maleSmokerX, maleSmokerY)

y_pred = regressor.predict(6.5)
y_pred = sc_y.inverse_transform(y_pred)

import matplotlib.pyplot as plt

plt.scatter(maleSmokerX, maleSmokerY, color = 'red')
plt.plot(maleSmokerX, regressor.predict(maleSmokerX), color = 'blue')
plt.show()

#from sklearn.model_selection import train_test_split as split

#xTrain, xTest, yTrain, yTest = split(maleSmokerX, maleSmokerY, test_size=0.22, random_state=0)



#plt.scatter(maleSmokerX, maleSmokerY)
#plt.show()