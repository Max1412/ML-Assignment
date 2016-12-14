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
"""
df.survived.value_counts().plot(kind='bar', alpha=0.5, title="Survival: 1 = survived, 0 = died")
plt.grid(b=True, which='major', axis='y')
#plt.show()

plt.scatter(df.survived, df.age, alpha=0.5)
plt.title("Survival by age: 1 = survived, 0 = died")
plt.grid(b=True, which='major', axis='y')
plt.yticks(range(0, 100, 10))
plt.xticks(range(0, 2))
#plt.show()

df.pclass.value_counts().plot(kind="barh", alpha=0.5, title="People per class")
plt.xticks(range(0, 700, 100))
plt.grid(b=True, which='major', axis="x")
#plt.show()

df_group = df.groupby(["survived", "pclass"])["pclass"].count().unstack("survived")
df_group.plot(kind="barh", stacked=True)
plt.title("People survived/died per class")
plt.xticks(range(0, 700, 100))
plt.grid(b=True, which='major', axis='x')
#plt.show()

# New
df_group = df.groupby(["survived", "sex"])["sex"].count().unstack("survived")
df_group.plot(kind="barh", stacked=True)
plt.title("People survived/died per sex")
plt.xticks(range(0, 700, 100))
plt.grid(b=True, which='major', axis='x')
#plt.show()

"""
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

"""
my_data=[['slashdot','USA','yes',18,'None'],
        ['google','France','yes',23,'Premium'],
        ['digg','USA','yes',24,'Basic'],
        ['kiwitobes','France','yes',23,'Basic'],
        ['google','UK','no',21,'Premium'],
        ['(direct)','New Zealand','no',12,'None'],
        ['(direct)','UK','no',21,'Basic'],
        ['google','USA','no',24,'Premium'],
        ['slashdot','France','yes',19,'None'],
        ['digg','USA','no',18,'None'],
        ['google','UK','no',18,'None'],
        ['kiwitobes','UK','no',19,'None'],
        ['digg','New Zealand','yes',12,'Basic'],
        ['slashdot','UK','no',21,'None'],
        ['google','UK','yes',18,'Basic'],
        ['kiwitobes','France','yes',19,'Basic']]
"""
df_t = df.head(10)
print(df_t)

print("")
tesst = df_t.values
print(len(tesst))
print("")
print(tesst)
print("")
print(tesst[0][4])
print("")

# Divides a set on a specific column. Can handle numeric or nominal values
def divideset(rows, column, value):
    # Make a function that tells us if a row is in the first group (true) or the second group (false)
    split_function = None
    if isinstance(value, int) or isinstance(value, float):  # check if the value is a number i.e int or float
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value

    # Divide the rows into two sets and return them
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)


print("")
print(len(divideset(tesst, 4, 0)))


#Divide Dataset dataframe by column name by value value
def divSet(dataframe, name, value):

    df1 = dataframe[dataframe[name] == value]
    df2 = dataframe[dataframe[name] > value]

    return df1, df2

print("")
print("")
print("")
da1, da2 = divSet(df_t, 'sex', 0)
print(da1)

def uniquecounts(dataframe):
   # val = dataframe.groupby('survived').count()
    val = dataframe['survived'].value_counts()
    return val

print("")
print("")
print("")
print(uniquecounts(da1))



# Entropy is the sum of p(x)log(p(x)) across all
# the different possible results
def entropy(dataframe):
   from math import log
   log2=lambda x:log(x)/log(2)
   results=uniquecounts(dataframe)
   # Now calculate the entropy
   ent=0.0
   for r in results.keys():
      p=float(results[r])/len(dataframe)
      ent=ent-p*log2(p)
   return ent

print("")
print("")
print("")
print(entropy(da1))


class decisionnode:
  def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
    self.col=col
    self.value=value
    self.results=results
    self.tb=tb
    self.fb=fb





"""
# Create counts of possible results (the last column of each row is the result)
def uniquecounts(rows):
   results={}
   for row in rows:
      # The result is the last column
      r=row[len(row)-1]
      if r not in results: results[r] = 0
      results[r]+=1
   return results
"""

def divide_set(dataframe, name):
	var_list = []
	for i in dataframe[name]:
		if i not in var_list:
			var_list.append(i)
	
	dataset_list = []
	for i in var_list:
		df = dataframe[dataframe[name] == i]
		dataset_list.append(df)

	return dataset_list
	
	'''
	T: whole dataset (before the split
	dfs: list of datasets (after the spilt)
	'''
def information_gain(T, dfs)
	information_gain = entropy(T)
	for i in dfs:
		information_gain -= (len(i)/len(T)) * entropy(i)
	
	return information_gain
	
# TODO add predict, fit methods according to sklearn
# maybe just "use" this node which automatically recursively builds the tree
# inside a classifier class?
class node:
	def __init__(self, dataset)
		self.dataset=dataset
		self.entropy=entropy(dataset)
		self.children=[]
		self.isleaf = is_leaf()
		if not isleaf:
			spilt(self)
		
	def is_leaf(self):
		return (entropy(self.dataset) == 0)
	
	def split(self)
		feature_names = dataframe[features]
		max_ig = 0
		result_dfs = []
		for feature in feature_names:
			pot_child_dfs = divide_set(dataframe, feature)
			ig = information_gain(self.dataset, pot_child_dfs)
			if ig > max_ig:
				result_dfs = pot_child_dfs
		for x in result_dfs:
			new_node = node(x)
			self.children.append(new_node)
			