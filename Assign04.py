#from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
#from sklearn import tree
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#%matplotlib inline

"""
Task 1: Preprocessing
"""

titanic_df = pd.read_csv('data.csv')
titanic_df.drop(titanic_df.columns[[0]], axis=1, inplace=True)
titanic_df.replace(' ?', np.nan, inplace=True)
titanic_df = titanic_df.dropna()

# Stripping whitespace
for i in titanic_df:
    titanic_df[i] = titanic_df[i].map(lambda x: x.lstrip(' ').rstrip(' ') if isinstance(x, str) else x)

plot_df = titanic_df.copy()

# Handling categorical values (replace by numerical values)
# Sex: Male = 0, Female = 1
titanic_df.replace('male', 0, inplace = True)
titanic_df.replace('female', 1, inplace = True)

# embarked: S = 0, C = 1, Q = 2
titanic_df.replace('S', 0, inplace = True)
titanic_df.replace('C', 1, inplace = True)
titanic_df.replace('Q', 2, inplace = True)

plot_df['survived'].replace(0, 'died', inplace=True)
plot_df['survived'].replace(1, 'survived', inplace=True)

# plot survival count
ax = sns.countplot(y="survived", data=plot_df)
ax.set(ylabel='Survival')
plt.show()

plot_df['survived'].replace('died', 0, inplace=True)
plot_df['survived'].replace('survived', 1, inplace=True)

# chance of survival by sex
survival_by_sex = plot_df[["sex", "survived"]].groupby(['sex'], as_index=False).mean()
ax = sns.barplot(x="sex", y="survived", data=survival_by_sex)
ax.set(xlabel="Sex", ylabel="Chance of survival")
plt.show()

# chance of survival by class
survival_by_class = plot_df[["pclass", "survived"]].groupby(['pclass'], as_index=False).mean()
ax = sns.barplot(x="pclass", y="survived", data=survival_by_class)
ax.set(xlabel="Class", ylabel="Chance of survival")
plt.show()

# survival by port of embarkation
survival_by_embarked = plot_df[["embarked", "survived"]].groupby(['embarked'], as_index=False).mean()
ax = sns.barplot(x="embarked", y="survived", data=survival_by_embarked)
ax.set(xlabel="Port of Embarkation", ylabel="Chance of survival")
plt.show()

plot_df['survived'].replace(0, 'died', inplace=True)
plot_df['survived'].replace(1, 'survived', inplace=True)

# plot age
sns.kdeplot(plot_df['age'].loc[plot_df['survived'] == 'survived'], shade=True, cut=0, label='survived')
sns.kdeplot(plot_df['age'].loc[plot_df['survived'] == 'died'], shade=True, cut=0, label='died')
plt.xlabel("Age")
plt.show()

# plot fare
sns.kdeplot(plot_df['fare'].loc[plot_df['survived'] == 'survived'], shade=True, cut=0, label='survived')
sns.kdeplot(plot_df['fare'].loc[plot_df['survived'] == 'died'], shade=True, cut=0, label='died')
plt.xlabel("Fare")
sns.plt.show()

# Age:
plot_df['age'] = plot_df['age'].astype(np.int64)
plot_df.loc[plot_df['age'] < 15, 'age'] = 0
plot_df.loc[(plot_df['age'] >= 15) & (plot_df['age'] <= 60), 'age'] = 1
plot_df.loc[plot_df['age'] > 60, 'age'] = 2
plot_df['age'].replace(0, "child", inplace=True)
plot_df['age'].replace(1, "middle-aged", inplace=True)
plot_df['age'].replace(2, "old", inplace=True)

titanic_df['age'] = titanic_df['age'].astype(np.int64)
titanic_df.loc[titanic_df['age'] < 15, 'age'] = 0
titanic_df.loc[(titanic_df['age'] >= 15) & (titanic_df['age'] <= 60), 'age'] = 1
titanic_df.loc[titanic_df['age'] > 60, 'age'] = 2

plot_df['survived'].replace('died', 0, inplace=True)
plot_df['survived'].replace('survived', 1, inplace=True)

# chance of survival by age
survival_by_age = plot_df[["age", "survived"]].groupby(['age'], as_index=False).mean()
ax = sns.barplot(x="age", y="survived", data=survival_by_age)
ax.set(xlabel="Age", ylabel="Chance of survival")
plt.show()

# grouped by age
ax = sns.factorplot(x="sex", y="survived", col="age", data=plot_df, kind="bar", ci=None, col_order=['child', 'middle-aged', 'old'])
ax.set_axis_labels("", "Chance of survival").set_titles("{col_name}").set(ylim=(0, 1)).despine(left=True)
plt.show()

# grouped by sex
ax = sns.factorplot(x="age", y="survived", col="sex", data=plot_df, kind="bar", ci=None, order=['child', 'middle-aged', 'old'])
ax.set_axis_labels("", "Chance of survival").set_titles("{col_name}").despine(left=True)
plt.show()

# grouped by class
ax = sns.factorplot(x="age", y="survived", col="pclass", data=plot_df, kind="bar", ci=None, order=['child', 'middle-aged', 'old'])
ax.set_axis_labels("", "Chance of survival").set_titles("Class {col_name}").despine(left=True)
plt.show()

print(titanic_df.head(50))