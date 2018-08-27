import pandas
from sklearn.cross_validation import train_test_split as split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix as matrix

data = pandas.read_csv('iris.data')

x = data.iloc[:, :4].values
y = data.iloc[:, 4].values

yLabelEncoder = LabelEncoder()
y = yLabelEncoder.fit_transform(y)

xTrain, xTest, yTrain, yTest = split(x, y, test_size = 1/3, random_state = 0)

classifier = LogisticRegression()
classifier.fit(xTrain, yTrain)

yPred = classifier.predict(xTest)

cm = matrix(yTest, yPred)