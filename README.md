# Titanic - Machine Learning from Disaster - Kaggle

This is a Kaggle's competicion

The challenge is use machine learning to create a model that predicts which passengers survived the Titanic shipwreck
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold

from sklearn.linear_model import LogisticRegression

%matplotlib inline
%pylab inline
def transformSex(valor):
    if valor == 'female':
        return 1 
    else: 
        return 0
# Data preparation
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train['Sex_bin'] = train['Sex'].map(transformSex)
test['Sex_bin'] = test['Sex'].map(transformSex)

variable = ['Sex_bin','Age']

x = train[variable].fillna(-1)
y = train['Survived']
# Model 01

Random Forest with 2 variables
result = []
kf = RepeatedKFold(n_splits=2, n_repeats=10, random_state=10)

for row_train, row_valid in kf.split(x):
    x_train, x_valid = x.iloc[row_train], x.iloc[row_valid]
    y_train, y_valid = y.iloc[row_train], y.iloc[row_valid]

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1,random_state=0)
    model.fit(x_train,y_train)

    p = model.predict(x_valid)

    acc = np.mean(y_valid == p)
    result.append(acc)  
np.mean(result)
pylab.hist(result)
# Model 02

Random Forest with 6 variables
variable = ['Sex_bin','Age','Pclass', 'SibSp', 'Parch', 'Fare']

x = train[variable].fillna(-1)
y = train['Survived']
result = []
kf = RepeatedKFold(n_splits=2, n_repeats=10, random_state=10)

for row_train, row_valid in kf.split(x):
    x_train, x_valid = x.iloc[row_train], x.iloc[row_valid]
    y_train, y_valid = y.iloc[row_train], y.iloc[row_valid]

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1,random_state=0)
    model.fit(x_train,y_train)

    p = model.predict(x_valid)

    acc = np.mean(y_valid == p)
    result.append(acc) 
np.median(result)
pylab.hist(result)
# Model 03

Logistic Regression with 6 variables
result = []
kf = RepeatedKFold(n_splits=2, n_repeats=7, random_state=10)

for row_train, row_valid in kf.split(x):
    x_train, x_valid = x.iloc[row_train], x.iloc[row_valid]
    y_train, y_valid = y.iloc[row_train], y.iloc[row_valid]

    model = LogisticRegression()
    
    model.fit(x_train,y_train)

    p = model.predict(x_valid)

    acc = np.mean(y_valid == p)
    result.append(acc) 
np.median(result)