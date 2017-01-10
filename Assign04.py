from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import pydotplus
#%matplotlib inline
sns.set_style('whitegrid')


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
titanic_df.replace('male', 0, inplace=True)
titanic_df.replace('female', 1, inplace=True)

# embarked: S = 0, C = 1, Q = 2
titanic_df.replace('S', 0, inplace=True)
titanic_df.replace('C', 1, inplace=True)
titanic_df.replace('Q', 2, inplace=True)

plot_df['survived'].replace(0, 'died', inplace=True)
plot_df['survived'].replace(1, 'survived', inplace=True)

# plot survival count
ax = sns.countplot(y="survived", data=plot_df)
ax.set(ylabel='Survival')
#plt.show()

plot_df['survived'].replace('died', 0, inplace=True)
plot_df['survived'].replace('survived', 1, inplace=True)

# chance of survival by sex
survival_by_sex = plot_df[["sex", "survived"]].groupby(['sex'], as_index=False).mean()
ax = sns.barplot(x="sex", y="survived", data=survival_by_sex)
ax.set(xlabel="Sex", ylabel="Chance of survival")
#plt.show()

# chance of survival by class
survival_by_class = plot_df[["pclass", "survived"]].groupby(['pclass'], as_index=False).mean()
ax = sns.barplot(x="pclass", y="survived", data=survival_by_class)
ax.set(xlabel="Class", ylabel="Chance of survival")
#plt.show()

# survival by port of embarkation
survival_by_embarked = plot_df[["embarked", "survived"]].groupby(['embarked'], as_index=False).mean()
ax = sns.barplot(x="embarked", y="survived", data=survival_by_embarked)
ax.set(xlabel="Port of Embarkation", ylabel="Chance of survival")
#plt.show()

plot_df['survived'].replace(0, 'died', inplace=True)
plot_df['survived'].replace(1, 'survived', inplace=True)

# plot age
sns.kdeplot(plot_df['age'].loc[plot_df['survived'] == 'survived'], shade=True, cut=0, label='survived')
sns.kdeplot(plot_df['age'].loc[plot_df['survived'] == 'died'], shade=True, cut=0, label='died')
plt.xlabel("Age")
#plt.show()

# plot fare
sns.kdeplot(plot_df['fare'].loc[plot_df['survived'] == 'survived'], shade=True, cut=0, label='survived')
sns.kdeplot(plot_df['fare'].loc[plot_df['survived'] == 'died'], shade=True, cut=0, label='died')
plt.xlabel("Fare")
#sns.plt.show()

# Age:
plot_df.loc[plot_df['age'] < 15, 'age'] = 0
plot_df.loc[(plot_df['age'] >= 15) & (plot_df['age'] <= 60), 'age'] = 1
plot_df.loc[plot_df['age'] > 60, 'age'] = 2
plot_df['age'].replace(0, "child", inplace=True)
plot_df['age'].replace(1, "middle-aged", inplace=True)
plot_df['age'].replace(2, "old", inplace=True)

plot_df['survived'].replace('died', 0, inplace=True)
plot_df['survived'].replace('survived', 1, inplace=True)

# chance of survival by age
survival_by_age = plot_df[["age", "survived"]].groupby(['age'], as_index=False).mean()
ax = sns.barplot(x="age", y="survived", data=survival_by_age)
ax.set(xlabel="Age", ylabel="Chance of survival")
#plt.show()

# grouped by age
ax = sns.factorplot(x="sex", y="survived", col="age", data=plot_df, kind="bar", ci=None, col_order=['child', 'middle-aged', 'old'])
ax.set_axis_labels("", "Chance of survival").set_titles("{col_name}").set(ylim=(0, 1)).despine(left=True)
#plt.show()

# grouped by sex
ax = sns.factorplot(x="age", y="survived", col="sex", data=plot_df, kind="bar", ci=None, order=['child', 'middle-aged', 'old'])
ax.set_axis_labels("", "Chance of survival").set_titles("{col_name}").despine(left=True)
#plt.show()

# grouped by class
ax = sns.factorplot(x="age", y="survived", col="pclass", data=plot_df, kind="bar", ci=None, order=['child', 'middle-aged', 'old'])
ax.set_axis_labels("", "Chance of survival").set_titles("Class {col_name}").despine(left=True)
#plt.show()

# create categorical dataframe
titanic_categorical = plot_df.copy()

titanic_categorical['survived'].replace(0, 'died', inplace=True)
titanic_categorical['survived'].replace(1, 'survived', inplace=True)

titanic_categorical['fare'] = titanic_categorical['fare']
titanic_categorical.loc[titanic_categorical['fare'] == 0, 'fare'] = 0
titanic_categorical.loc[(titanic_categorical['fare'] > 0) & (titanic_categorical['fare'] <= 32), 'fare'] = 1
titanic_categorical.loc[(titanic_categorical['fare'] > 32) & (titanic_categorical['fare'] < 260), 'fare'] = 2
titanic_categorical.loc[titanic_categorical['fare'] >= 260, 'fare'] = 3
titanic_categorical['fare'].replace(0, "worker", inplace=True)
titanic_categorical['fare'].replace(1, "low-range", inplace=True)
titanic_categorical['fare'].replace(2, "mid-range", inplace=True)
titanic_categorical['fare'].replace(3, "high-range", inplace=True)

titanic_numeric = titanic_df.copy()
titanic_numeric.loc[titanic_numeric['age'] < 15, 'age'] = 0
titanic_numeric.loc[(titanic_numeric['age'] >= 15) & (titanic_numeric['age'] <= 60), 'age'] = 1
titanic_numeric.loc[titanic_numeric['age'] > 60, 'age'] = 2
titanic_numeric.loc[titanic_numeric['fare'] == 0, 'fare'] = 0
titanic_numeric.loc[(titanic_numeric['fare'] > 0) & (titanic_numeric['fare'] <= 32), 'fare'] = 1
titanic_numeric.loc[(titanic_numeric['fare'] > 32) & (titanic_numeric['fare'] < 260), 'fare'] = 2
titanic_numeric.loc[titanic_numeric['fare'] >= 260, 'fare'] = 3

# rm sibsp and parch
titanic_numeric.drop(titanic_numeric.columns[[4, 5]], axis=1, inplace=True)
titanic_df.drop(titanic_df.columns[[4, 5]], axis=1, inplace=True)
titanic_categorical.drop(titanic_categorical.columns[[4, 5]], axis=1, inplace=True)


"""
Task 3
"""

"""
- splitting feature
- gini / information gain
- threshold
- children
- leaf = boolean
- decision
- parent?
- ???
"""


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, measure_function="info_gain", demo_param='demo'):
        self.demo_param = demo_param
        self.calc_measure = None
        self.root = None
        if measure_function == "info_gain":
            self.calc_measure = self.information_gain
            # elif measure_function == "gini":
            #     self.calc_measure = self.gini

    def fit(self, x, y):
        self.root = DecisionNode(x, y, self.calc_measure)
        # Return the estimator
        return self

    def predict(self, x):
        y = self.root.traverse(x)
        return y

    @staticmethod
    def information_gain(data, split_data):
        def log2(x):
            from math import log
            return log(x)/log(2)

        def entropy(t):
            results = t.value_counts()
            # Now calculate the entropy
            ent = 0
            for r in results.keys():
                p = float(results[r])/len(t)
                ent = ent-p*log2(p)
            return ent

        gain = entropy(data)
        for d in split_data:
            gain -= (len(d)/len(data)) * entropy(d)
        return gain


class DecisionNode:
    def __init__(self, x, y, calc_measure):
        self.attributes = x.columns
        self.data = x
        self.results = y
        self.children = None
        self.decision_attribute = y.name
        self.calc_measure = calc_measure
        self.measure_of_impurity = 0
        self.decision = y.value_counts().idxmax()
        self.split_attribute = ""
        self.decide()
        # print("Node: {}".format(self.split_attribute))
        # print("Decision: {}".format(self.decision))
        # print(x)
        # print(y)

    def divide_set(self, attribute):
        distinct_values = self.data[attribute].unique()
        datasets = []
        for v in distinct_values:
            datasets.append(self.results[self.data[attribute] == v])
        return datasets

    def decide(self):
        max_measure_of_impurity = 0
        best_attribute = ""
        for attribute in self.attributes:
            measure_of_impurity = self.calc_measure(self.results, self.divide_set(attribute))
            if max_measure_of_impurity < measure_of_impurity:
                max_measure_of_impurity = measure_of_impurity
                best_attribute = attribute
        self.split_attribute = best_attribute
        if max_measure_of_impurity > 0:
            self.split(best_attribute)

    def split(self, attribute):
        distinct_values = self.data[attribute].unique()
        if len(distinct_values) > 1:
            self.children = {}
            for v in distinct_values:
                child_node = DecisionNode(self.data[self.data[attribute] == v], self.results[self.data[attribute] == v],
                                          self.calc_measure)
                self.children[v] = child_node

    def traverse(self, x):
        results = np.array([])
        if self.children is None:
            results = np.append(results, [self.decision for x_i in range(len(x))])
        else:
            distinct_values = x[self.split_attribute].unique()
            if len(distinct_values) > 0:
                for v in distinct_values:
                    results = np.append(results, self.children[v].traverse(x[x[self.split_attribute] == v]))
            else:
                results = np.append(results, [self.decision for x_i in range(len(x))])
        return results
        # if self.children is None:
        #     return self.decision
        # results = []
        # for _, x_i in x.iterrows():
        #     print(x_i)
        #     results.append(
        #         self.children[
        #             x_i[self.split_attribute]
        #         ].traverse(x_i))
        # return np.array(results)


"""
Task 2
"""

features = list(titanic_df.columns[1:])
print(features)
titanic_y = titanic_df["survived"]
titanic_X = titanic_df[features]

titanic_x_train, titanic_x_test, titanic_y_train, titanic_y_test = train_test_split(titanic_X, titanic_y, test_size=0.2,
                                                                                    random_state=1, stratify=titanic_y)

dt = DecisionTreeClassifier(min_samples_leaf=3, min_impurity_split=0.02)
dt.fit(titanic_x_train, titanic_y_train)
titanic_res = dt.predict(titanic_x_test)

print("Initial Values")
print(accuracy_score(titanic_y, dt.predict(titanic_X)))
print("Decision Tree Classifier")
print(accuracy_score(titanic_y_test, titanic_res))
print(classification_report(titanic_y_test, titanic_res))

dot_data = tree.export_graphviz(dt, out_file=None,
                                feature_names=features,
                                class_names=["died", "survived"],
                                filled=True, rounded=True,
                                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('tree.png')

# our implementation

features = list(titanic_numeric.columns[1:])
print(features)
titanic_y = titanic_numeric["survived"]
titanic_X = titanic_numeric[features]

titanic_x_train, titanic_x_test, titanic_y_train, titanic_y_test = train_test_split(titanic_X, titanic_y, test_size=0.2,
                                                                                    random_state=1, stratify=titanic_y)

dt = DecisionTree()
dt.fit(titanic_x_train, titanic_y_train)
titanic_res = dt.predict(titanic_x_test)

print("Initial Values")
print(accuracy_score(titanic_y, dt.predict(titanic_X)))
print("Decision Tree Classifier")
print(accuracy_score(titanic_y_test, titanic_res))
print(classification_report(titanic_y_test, titanic_res))

# numeric
features = list(titanic_numeric.columns[1:])
print(features)
titanic_y = titanic_numeric["survived"]
titanic_X = titanic_numeric[features]

titanic_x_train, titanic_x_test, titanic_y_train, titanic_y_test = train_test_split(titanic_X, titanic_y, test_size=0.2,
                                                                                    random_state=1, stratify=titanic_y)

dt = DecisionTreeClassifier()
dt.fit(titanic_x_train, titanic_y_train)

titanic_res = dt.predict(titanic_x_test)

print("Categorical Values via numeric representation")
print(accuracy_score(titanic_y, dt.predict(titanic_X)))
print("Decision Tree Classifier")
print(accuracy_score(titanic_y_test, titanic_res))
print(classification_report(titanic_y_test, titanic_res))

dot_data = tree.export_graphviz(dt, out_file=None, feature_names=features, class_names=["died", "survived"],
                                filled=True, rounded=True, special_characters=True)


graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('tree_numeric.png')

"""
 Image(graph.create_png())          in iPython
"""

