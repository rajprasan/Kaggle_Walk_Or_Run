
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

wr=pd.read_csv('dataset.csv')
wr.head()
wr.columns
#sns.heatmap(vc.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#sns.pairplot(vc)

X= wr[['wrist', 'acceleration_x',
       'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z' ]]
#, 'sd', 'median','Q25', 'Q75', 'modindx','dfrange'
y=wr['activity']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print('logistic regression report')
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

# conducting decision tress
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)

pred = dtree.predict(X_test)
print('decision tree report')
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

# conducting random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print('Random forest report')
print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))


