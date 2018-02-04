
# coding: utf-8

# In[12]:

from sklearn import tree

features = [[140 , 1],[120 , 1],[130 , 1],[138 , 1],[150 , 0],[162 , 0],[178 , 0],[170 , 0]]
labels = [1,1,1,1,0,0,0,0]

clf = tree.DecisionTreeClassifier()
clf.fit(features,labels)

predict_ = [[122,0]]
if(clf.predict(predict_)[0] == 0):
    print("Orange")
else:
    print("Apple")

    


# In[25]:

from sklearn.datasets import load_iris

iris_ = load_iris()

from sklearn import tree
import numpy as np

test_idx = [0, 50, 100]

#Training Data
train_target = np.delete(iris_.target,test_idx)
train_data = np.delete(iris_.data, test_idx,axis=0)

#Testing Data
test_target = iris_.target[test_idx]
test_data = iris_.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)


print(clf.predict(test_data))
    


# In[1]:

import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs =500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height,lab_height],stacked=True,color=['r','b'])
plt.show()


# In[72]:

from sklearn import datasets
iris = datasets.load_iris()

X_ = iris.data
Y_= iris.target

from sklearn.cross_validation import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X_,Y_,test_size = .5)

#from sklearn import tree
#clf = tree.DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()

clf.fit(X_train,Y_train)

#print(X_test)

print(clf.predict([[6,2,4,1]]))

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,clf.predict(X_test))*100)

