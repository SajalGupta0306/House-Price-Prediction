# importing libraries
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# loading iris data set
irisData = datasets.load_iris()

# printing description, features, labels
desc = irisData.DESCR
features = irisData.data
labels = irisData.target

# training the model
model = KNeighborsClassifier()
model.fit(features, labels)

predictedValue = model.predict([[3, 1, 2, 3]])
print(predictedValue)

