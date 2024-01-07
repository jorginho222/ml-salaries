import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# df = pd.read_csv('../data/salaries.csv')
df = pd.read_csv('../data/salaries-cities.csv')

x = df.iloc[:, :-1]  # get all rows with all columns except the last one #
y = df.iloc[:, -1]   # get all rows with only the last column

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)
# error = y_pred - y_test
# print(error)

r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2} ({r2:.2%})")

# salaries = model.predict([[11, 1], [11, 2], [12, 1], [12, 2]])  # pass specific x values
# print(salaries)

# plt.scatter(x_test, y_test)
# plt.plot(x_test, y_pred, color="yellow")
# plt.show()

