# -*- coding: utf-8 -*-
"""
Tony Silva

Condensed Titanic Practice
"""
import pandas as pd
import numpy
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv("C:/Users/Anthony Silva/silvat/titanic/train.csv")
train = train.set_index("PassengerId")
test = pd.read_csv("C:/Users/Anthony Silva/silvat/titanic/test.csv")
test = test.set_index("PassengerId")
titanic = train.loc[:, "Pclass":]
titanic = titanic.append(test)
titanic = pd.concat([titanic, train["Survived"]], axis=1)

titanic["Cabin"] = titanic["Cabin"].fillna("Z")
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].mean())
titanic["Fare"] = titanic["Fare"].round(2)
df_age = pd.get_dummies(titanic["Sex"])
titanic = pd.concat([titanic, df_age], axis=1)
del df_age


# Create Feature Vectors for Pclass variable
# Commented out because class had order to it
#df_pclass = pd.get_dummies(titanic["Pclass"])
#df_pclass.columns = ["pclass1","pclass2","pclass3"]
#titanic = pd.concat([titanic, df_pclass], axis=1)
#del df_pclass

# Create Feature Vectors for Embarked variable
df_embark = pd.get_dummies(titanic["Embarked"])
titanic = pd.concat([titanic, df_embark], axis=1)
del df_embark
#
## Create new features for Family Size
titanic["famsize"] = titanic["SibSp"] + titanic["Parch"] + 1
## Create a new feature for Title, Cabin Letter, Price of a ticket
## Title is Mr, Mrs, Miss ,etc
## Cabin Letter is the beginning letter of the cabin, if null then N
## Ticket Count is the number of tickets in the data set. 
tickets = titanic["Ticket"].value_counts()
for i in titanic.itertuples():
    title = i[2].replace(' ','').replace('.',',').split(',')
    cl = str(i[10])
    cl = cl.upper()
    if cl[0] == "C": 
        cl = "CC"
    else:
        cl = cl[0]
    ticket = tickets[i[7]]
    titanic.loc[i[0], "title"] = title[1]
    titanic.loc[i[0], "CL"] = cl
    titanic.loc[i[0], "ticket_count"] = ticket
del title, ticket, tickets, cl, i

titanic["price"] = titanic.Fare / titanic.ticket_count
titanic["price"] = titanic["price"].round(2)
titanic["Age"] = titanic["Age"].round()

df_title = pd.get_dummies(titanic["title"])
titanic = pd.concat([titanic, df_title], axis=1)
del df_title

df_CL = pd.get_dummies(titanic["CL"])
titanic = pd.concat([titanic, df_CL], axis=1)
del df_CL

df_pclass = pd.get_dummies(titanic["Pclass"])
df_pclass.columns = ["pclass1","pclass2","pclass3"]
titanic = pd.concat([titanic, df_pclass], axis=1)
del df_pclass

titanic = titanic.drop(["Pclass","Embarked", "Fare", "Sex", "title", "Ticket", "CL", "Cabin", "Name"], axis=1)


# Create a Linear Regression Model to predict the Age of the null Ages
age_pred = LinearRegression()
train = titanic.loc[titanic.Age.notnull(), :]
train = train.drop("Survived", axis=1)
train_X = train.drop("Age", axis=1)
train_Y = train["Age"]
age_pred.fit(train_X, train_Y)
#for i in range(len(list(train_X.columns.values))):
#    print(list(train_X.columns.values)[i], list(age_pred.coef_)[i])
#print(age_pred.score(train_X, train_Y))
test = titanic.loc[titanic.Age.isnull(),:]
test = test.drop("Survived", axis=1)
test_X = test.drop("Age", axis=1)
titanic.loc[titanic.Age.isnull(),"Age"] = age_pred.predict(test_X)
del test,test_X,age_pred

# Create an age group feature.
titanic.loc[titanic.Age < 18, "age_grp"] = 0
titanic.loc[(titanic.Age >= 18) & (titanic["Age"] < 60), "age_grp"] = 1
titanic.loc[titanic.Age >= 60, "age_grp"] = 2
#titanic["age_grp"] = titanic["age_grp"].astype(int)
titanic["Survived"] = titanic["Survived"].astype('category')

train = titanic.loc[titanic.Survived.notnull(), :]
train, vald = train_test_split(train, test_size=.3)
test = titanic.loc[titanic.Survived.isnull(), :]
test = test.drop("Survived", axis=1)
train_X = train.drop("Survived", axis=1)
train_Y = train["Survived"]
vald_X = vald.drop("Survived", axis=1)
actual = vald["Survived"]

#Logistic Regression Model, accuracy of .77512
clf = LogisticRegression()
clf.fit(train_X, train_Y)
predictions = clf.predict(vald_X)
print("Logisitic Regression Model")
print(clf.score(vald_X, actual))
print(confusion_matrix(actual, predictions, labels=[0,1]))

# Random Forest Classifier, accuracy of .79426 with splitting PClass out.
# .78 accuracy with not splitting out PClass.
clf = RandomForestClassifier(n_estimators=500,criterion="entropy", min_samples_leaf=4)
clf.fit(train_X, train_Y)
predictions = clf.predict(vald_X)
print("Random Forest Classifier")
print(clf.score(vald_X, actual))
print(confusion_matrix(actual, predictions, labels=[0,1]))

results = pd.Series(clf.predict(test), name="Survived")
results = results.astype(int)
PassengerId = pd.Series(test.index.values, name="PassengerId")
results = pd.concat([PassengerId, results], axis=1)
results = results.set_index("PassengerId")
results.to_csv("submission_2.csv")