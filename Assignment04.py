from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

"""
Task 1: Preprocessing
"""

df = pd.read_csv('data.csv')
# TODO: What is the first column? Maybe names or sth?
# TODO: The very first originates from pandas...
df.replace(' ?', np.nan, inplace=True)
df = df.dropna()

# Stripping whitespace
for i in df:
    df[i] = df[i].map(lambda x: x.lstrip(' ').rstrip(' ') if isinstance(x, str) else x)

# Handling categorical values (replace by numerical values)
# Sex: Male = 0, Female = 1
df.replace('male', 0, inplace = True)
df.replace('female', 1, inplace = True)

# embarked: S = 0, C = 1, Q = 2
df.replace('S', 0, inplace = True)
df.replace('C', 1, inplace = True)
df.replace('Q', 2, inplace = True)

#print(df.head(50))

"""
Observations about the data (https://www.kaggle.com/c/titanic/data);
    - Why are the attributes 'sibsp' and 'parch' not combined to, say, 'relatives in board'

Possible interesting result:
    - Percentage of survived children/adults

Expectations of the prediction task (which features might contribute?)
    - Sex and Age: 'Children and females first'
    - (maybe) SES -> pclass, rich people might be able to get off first (upper floor cabins)
    - POE might not be so relevant (cabin location or anything else relevant might not depend on this)
    - We do not have the 'cabin' attribute (as opposed to the original kaggle data) though it might be interesting
"""


"""
Plots:
"""

df.survived.value_counts().plot(kind='bar', alpha=0.5, title="Survival: 1 = survived, 0 = died")
plt.grid(b=True, which='major', axis='y')
plt.show()

plt.scatter(df.survived, df.age, alpha=0.5)
plt.title("Survival by age: 1 = survived, 0 = died")
plt.grid(b=True, which='major', axis='y')
plt.yticks(range(0, 100, 10))
plt.xticks(range(0, 2))
plt.show()

df.pclass.value_counts().plot(kind="barh", alpha=0.5, title="People per class")
plt.xticks(range(0, 700, 100))
plt.grid(b=True, which='major', axis="x")
plt.show()

df_group = df.groupby(["survived", "pclass"])["pclass"].count().unstack("survived")
df_group.plot(kind="barh", stacked=True)
plt.title("People survived/died per class")
plt.xticks(range(0, 700, 100))
plt.grid(b=True, which='major', axis='x')
plt.show()


"""
Task 2
"""

features = list(df.columns[2:9])
#print(features)
y = df["survived"]
X = df[features]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1 )

dt = DecisionTreeClassifier()
dt.fit(X, y)

res = dt.predict(x_test)
# TODO Bugfixing?
print(accuracy_score(y, dt.predict(X)))
print("Decision Tree Classifier")
print(accuracy_score(y_test, res))

########## Own Code ########
