import pandas

data = pandas.read_csv('insurance.csv')

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

le = LabelEncoder()

x[:, 1] = le.fit_transform(x[:, 1])
x[:, 4] = le.fit_transform(x[:, 4])
x[:, 5] = le.fit_transform(x[:, 5])

ohe = OneHotEncoder(categorical_features=[5])
x = ohe.fit_transform(x).toarray()

from sklearn.model_selection import train_test_split as split

xTrain, xTest, yTrain, yTest = split(x, y, test_size=0.22, random_state=0)

ssx = StandardScaler()
ssy = StandardScaler()
xTrain = ssx.fit_transform(xTrain)
xTest = ssx.transform(xTest)
yTrain = ssy.fit_transform(yTrain.reshape(-1, 1))
yTest = ssy.fit_transform(yTest.reshape(-1, 1))


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xTrain, yTrain)

yPred = regressor.predict(xTest)

regressor.score(xTest,yTest)

import statsmodels.formula.api as sm
import numpy as np

x = np.append(arr = np.ones((1338, 1)).astype(int), values = x, axis = 1)
xOpt = x[:, [0, 1, 2, 3, 4, 5, 7, 8, 9]]
regressor_OLS = sm.OLS(endog = y, exog = xOpt).fit()
regressor_OLS.summary()

