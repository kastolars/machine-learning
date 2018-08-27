import numpy
import matplotlib.pyplot
import pandas
from sklearn.model_selection import train_test_split as split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

data = pandas.read_csv('iris.data')

x = data.iloc[:, :4]
y = data.iloc[:, 4]

yLabelEncoder = LabelEncoder()
y = yLabelEncoder.fit_transform(y)

xTrain, xTest, yTrain, yTest = split(x, y, test_size = 1/3, random_state = 0)

regressor = LinearRegression()
regressor.fit(xTrain, yTrain)

yPrediction = regressor.predict(xTest)

yPredRounded = list(map(lambda y : int(round(y)), yPrediction))

res = yTest == yPredRounded

numPredictions = len(yPredRounded)

res = list(res)

numTrue = res.count(True)

accuracy = float(numTrue/numPredictions) * 100

print(accuracy)
