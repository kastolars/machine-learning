import pandas
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, Imputer
from sklearn.model_selection import train_test_split as split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as matrix

data = pandas.read_csv('adult.data')

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

#imputer = Imputer(missing_values = '?', strategy = 'most_frequent', axis = 0)
#imputer = imputer.fit(x[:, 1])
#x[:, 1] = imputer.transform(x[:, 1])


myLabelEncoder = LabelEncoder()
x[:, 1] = myLabelEncoder.fit_transform(x[:, 1])
x[:, 3] = myLabelEncoder.fit_transform(x[:, 3])
x[:, 5] = myLabelEncoder.fit_transform(x[:, 5])
x[:, 6] = myLabelEncoder.fit_transform(x[:, 6])
x[:, 7] = myLabelEncoder.fit_transform(x[:, 7])
x[:, 8] = myLabelEncoder.fit_transform(x[:, 8])
x[:, 9] = myLabelEncoder.fit_transform(x[:, 9])
x[:, 13] = myLabelEncoder.fit_transform(x[:, 13])

myHotEncoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
x = myHotEncoder.fit_transform(x).toarray()
y = myLabelEncoder.fit_transform(y)

xTrain, xTest, yTrain, yTest = split(x, y, test_size = 1/3, random_state = 0)

classifier = LogisticRegression()
classifier.fit(xTrain, yTrain)

yPred = classifier.predict(xTest)

cm = matrix(yTest, yPred)
