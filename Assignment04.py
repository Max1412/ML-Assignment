from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('data.csv')
#TODO: What is the first column? Maybe names or sth?
#TODO: The very first originates from pandas...
df.replace(' ?', np.nan, inplace=True)
df = df.dropna()


for i in df:
    df[i] = df[i].map(lambda x: x.lstrip(' ').rstrip(' ') if isinstance(x, str) else x)
#Sex: Male = 0, Female = 1
df.replace('male', 0, inplace = True)
df.replace('female', 1, inplace = True)
#embarked: S = 0, C = 1, Q = 2
df.replace('S', 0, inplace = True)
df.replace('C', 1, inplace = True)
df.replace('Q', 2, inplace = True)


#print(df.head(50))


features = list(df.columns[2:9])
#print(features)
y = df["survived"]
X = df[features]
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X, y)
