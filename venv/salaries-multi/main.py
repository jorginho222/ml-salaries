import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/salaries-2023.csv')
df = df.dropna()

print(df.head())
print(df.shape)
df.info()