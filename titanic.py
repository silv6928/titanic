# -*- coding: utf-8 -*-
"""
Tony Silva
Machine Learning Practice
"""
import pandas as pd
import numpy
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
'''
Data Cleaning Operations
-Loading Data
-Fix missing values
-Create feature vectors
Visual Analysis
Predictive Model Creation
'''
# Load Data
titanic = pd.read_csv("C:/Users/Anthony Silva/silvat/titanic/train.csv")
titanic = titanic.set_index("PassengerId")
# Fix Missing Values
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].mean())
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic["Cabin"] = titanic["Cabin"].fillna("Z")
# Create Feature Vectors for Sex variable
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
    title = i[3].replace(' ','').replace('.',',').split(',')
    cl = str(i[10])
    cl = cl.upper()
    if cl[0] == "C": 
        cl = "CC"
    else:
        cl = cl[0]
    ticket = tickets[i[8]]
    titanic.loc[i[0], "title"] = title[1]
    titanic.loc[i[0], "CL"] = cl
    titanic.loc[i[0], "ticket_count"] = ticket
del title
del ticket
del tickets
del cl
del i
## Price is Fare / Ticket Count, some tickets were purchased in pairs for a family
## Gets the price per person, so as not to inflate the amount paid by person.
titanic["price"] = titanic.Fare / titanic.ticket_count

##print(len(titanic["title"].unique())) #17 unique titles
df_title = pd.get_dummies(titanic["title"])
# Combine the various versions of Miss
titanic = pd.concat([titanic, df_title], axis=1)
del df_title

df_CL = pd.get_dummies(titanic["CL"])
titanic = pd.concat([titanic, df_CL], axis=1)
del df_CL


#titanic = titanic.drop(["Cabin", "Name"], axis=1)
# Create an Age Group Feature
# 0 if a child, 1 if an adult, 2 if a senior.
titanic.loc[titanic.Age < 18, "age_grp"] = 0
titanic.loc[(titanic.Age >= 18) & (titanic["Age"] < 60), "age_grp"] = 1
titanic.loc[titanic.Age >= 60, "age_grp"] = 2
titanic["age_grp"] = titanic["age_grp"].astype(int)
titanic["Survived"] = titanic["Survived"].astype('category')


# Data Exploration
# Which variables help predict survival better
# More people died then survived.
'''
plt.figure()
sns.countplot(x="Survived", data=titanic)
plt.figure()
sns.countplot(x="age_grp", hue="Survived", data=titanic)
# If embarking from S, you were more likely to not survive.
plt.figure()
sns.countplot(x="Embarked", hue="Survived", data=titanic)
# Traveling by yourself made you more likely to not survive
plt.figure()
sns.countplot(x="famsize", hue="Survived", data=titanic)
# Being in third class made it more likely to not survive
plt.figure()
sns.countplot(x="Pclass", hue="Survived", data=titanic)
'''

### Generate a Predictive Model ###
# Drop out unused columns for Machine Learning.
titanic = titanic.drop(["Embarked", "Sex", "title",
                        "Ticket", "CL", "Cabin", "Name"], axis=1)

predictors = list(titanic.columns.values)[1:]
target = list(titanic.columns.values)[0]

# Create training and test sets at 70% train, 30% test
# 624 values in training and 267 in test based on split.
train, test = train_test_split(titanic, test_size=0.3)

clf = LogisticRegression()
clf.fit(train[predictors], train[target])
predictions = clf.predict(test[predictors])
actual = test[target]
print("Accuracy for LogReg: ", clf.score(test[predictors], test[target]))
print(classification_report(actual, predictions))


# Considerations: Create Ticket Featuress for all of the different tickets
del test

'''
Create Test Predictions for Kaggle
'''
test = pd.read_csv("C:/Users/Anthony Silva/silvat/titanic/test.csv")
test = test.set_index("PassengerId")
# Fix Missing Values
test["Age"] = test["Age"].fillna(test["Age"].mean())
test["Age"] = test.Age.round()
test["Fare"] = test["Fare"].fillna(test["Fare"].mean())
test["Fare"] = test.Fare.round(2)
test["Embarked"] = test["Embarked"].fillna("S")
test["Cabin"] = test["Cabin"].fillna("Z")
# Create Feature Vectors for Sex variable
df_age = pd.get_dummies(test["Sex"])
test = pd.concat([test, df_age], axis=1)
del df_age



# Create Feature Vectors for Embarked variable
df_embark = pd.get_dummies(test["Embarked"])
test = pd.concat([test, df_embark], axis=1)

del df_embark
#
## Create new features for Family Size
test["famsize"] = test["SibSp"] + test["Parch"] + 1
## Create a new feature for Title, Cabin Letter, Price of a ticket
## Title is Mr, Mrs, Miss ,etc
## Cabin Letter is the beginning letter of the cabin, if null then N
## Ticket Count is the number of tickets in the data set. 
tickets = test["Ticket"].value_counts()
for i in test.itertuples():
    title = i[2].replace(' ','').replace('.',',').split(',')
    cl = str(i[9])
    cl = cl.upper()
    if cl[0] == "C": 
        cl = "CC"
    else:
        cl = cl[0]
    ticket = tickets[i[7]]
    test.loc[i[0], "title"] = title[1]
    test.loc[i[0], "CL"] = cl
    test.loc[i[0], "ticket_count"] = ticket
del title
del ticket
del tickets
del cl
del i
## Price is Fare / Ticket Count, some tickets were purchased in pairs for a family
## Gets the price per person, so as not to inflate the amount paid by person.
test["price"] = test.Fare / test.ticket_count

##print(len(test["title"].unique())) #17 unique titles
df_title = pd.get_dummies(test["title"])
test = pd.concat([test, df_title], axis=1)
del df_title

df_CL = pd.get_dummies(test["CL"])
test = pd.concat([test, df_CL], axis=1)
del df_CL


# Create an Age Group Feature
# 0 if a child, 1 if an adult, 2 if a senior.
test.loc[test.Age < 18, "age_grp"] = 0
test.loc[(test.Age >= 18) & (test["Age"] < 60), "age_grp"] = 1
test.loc[test.Age >= 60, "age_grp"] = 2
test["age_grp"] = test["age_grp"].astype(int)


### Generate a Predictive Model ###
# Drop out unused columns for Machine Learning.
test = test.drop(["Embarked", "Ticket", "Sex", "title",
                         "CL", "Cabin", "Name"], axis=1)
test
# Return the Columns in the Training set not in the test set
cols = list(set(titanic.columns.values) - set(test.columns.values))
for i in cols:
    test[i] = 0
results = pd.Series(clf.predict(test[predictors]), name="Survived")
PassengerId = pd.Series(test.index.values, name="PassengerId")
results = pd.concat([PassengerId, results], axis=1)
results = results.set_index("PassengerId")

#results.to_csv("submission_1.csv")