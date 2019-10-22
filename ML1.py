import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetesData = datasets.load_diabetes()
# gets second feature and makes a new column for that
# diabetesData_X_Axis = diabetesData.data[:, np.newaxis, 2]
diabetesData_X_Axis = diabetesData.data

diabetesData_X_Axis_train = diabetesData_X_Axis[:-30]  # get training data for last 30 values
diabetesData_X_Axis_test = diabetesData_X_Axis[-30:]   # get test data for first 30 values

diabetesData_Y_Axis_train = diabetesData.target[:-30]  # get corresponding label data for last 30 values
diabetesData_Y_Axis_test = diabetesData.target[-30:]   # get corresponding label data for first 30 values

# build a linear regression model
model = linear_model.LinearRegression()
# fitting the model on X axis and y axis data
model.fit(diabetesData_X_Axis_train, diabetesData_Y_Axis_train)
# predicting the model on test data and getting the values on y axis
diabetesData_Y_Axis_predicted = model.predict(diabetesData_X_Axis_test)

# mean squared error is the difference of actual - predicted values and square of that and average of the values
print("Mean squared error is: ", mean_squared_error(diabetesData_Y_Axis_test, diabetesData_Y_Axis_predicted))
print("Weights are ", model.coef_)  #weights are w1,w2,....wn
print("Intercept is", model.intercept_)   # intercept is W0

# chart is based on y=w0 + w1.x1 where w0 is the intercept and w1 is the weight and x1 is the value of feature
# y is the value of the label
# the line can be drawn only with respect to one feature
# plt.scatter(diabetesData_X_Axis_test, diabetesData_Y_Axis_test)
# plt.plot(diabetesData_X_Axis_test, diabetesData_Y_Axis_predicted)
# plt.show()
#



