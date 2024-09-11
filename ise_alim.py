import numpy as np
import pandas as pd
from sklearn import tree

df = pd.read_csv("DecisionTreesClassificationDataSet.csv")

duzeltme_mapping = {'Y': 1, 'N': 0}
df['IseAlindi'] = df['IseAlindi'].map(duzeltme_mapping)
df['SuanCalisiyor?'] = df['SuanCalisiyor?'].map(duzeltme_mapping)
df['Top10 Universite?'] = df['Top10 Universite?'].map(duzeltme_mapping)
df['StajBizdeYaptimi?'] = df['StajBizdeYaptimi?'].map(duzeltme_mapping)

duzeltme_mapping_egitim = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Egitim Seviyesi'] = df['Egitim Seviyesi'].map(duzeltme_mapping_egitim)
y = df['IseAlindi']
X = df.drop(['IseAlindi'], axis = 1)

#Decision trees
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

print(clf.predict([[5, 1, 3, 0, 0, 0]]))
print(clf.predict([[2, 0, 7, 0, 1, 0]]))