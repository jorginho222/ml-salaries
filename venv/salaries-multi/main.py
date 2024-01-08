import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/salaries-2023.csv')
df = df.dropna()  # drop null values

# print(df.head())
allowed_languages = ['php', 'js', '.net', 'java']
df = df[df['language'].isin(allowed_languages)]

# the cities come from free-form text field and not a choice from a dropdown. So we make a list of possible users input
vilnius_names = ['Vilniuj', 'Vilniua', 'VILNIUJE', 'VILNIUS', 'vilnius', 'Vilniuje']
condition = df['city'].isin(vilnius_names)
df.loc[condition, 'city'] = 'Vilnius'

kaunas_names = ['KAUNAS', 'kaunas', 'Kaune']
condition = df['city'].isin(kaunas_names)
df.loc[condition, 'city'] = 'Kaunas'

allowed_cities = ['Vilnius', 'Kaunas']
df = df[df['city'].isin(allowed_cities)]
# print(df.shape)

df_sorted = df.sort_values(by='salary', ascending=False)
# print(df_sorted.head(20))

df = df[df['salary'] <= 6000 ]
# print(df.shape)
#
# x = df.iloc[:, -2:-1]
# y = df.iloc[:, -1].values
# plt.xlabel('Years of experience')
# plt.ylabel('Salary')
# plt.scatter(x, y)
# plt.show()

one_hot = pd.get_dummies(df['language'], prefix='lang')
df = df.join(one_hot)
df = df.drop('language', axis=1)

one_hot = pd.get_dummies(df['city'], prefix='city')
df = df.join(one_hot)
df = df.drop('city', axis=1)

print(df.head(10))
sns.heatmap(df.corr(), annot=True)
plt.show()

x = df.iloc[:, 0:2].values  # we take only years and level as x values
y = df.iloc[:, 2].values    # we take the salary
# print(x[0:5])
# print(y[0:5])


