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
'''
Data Cleaning Operations
-Loading Data
-Fix missing values
-Create feature vectors
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
df_pclass = pd.get_dummies(titanic["Pclass"])
df_pclass.columns = ["pclass1","pclass2","pclass3"]
titanic = pd.concat([titanic, df_pclass], axis=1)
del df_pclass

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
titanic.price = titanic.Fare / titanic.ticket_count

##print(len(titanic["title"].unique())) #17 unique titles
#df_title = pd.get_dummies(titanic["title"])
#titanic = pd.concat([titanic, df_title], axis=1)
#del df_title

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
plt.figure()
sns.countplot(x="Survived", data=titanic)
plt.figure()
sns.countplot(x="age_grp", hue="Survived", data=titanic)
plt.figure()
sns.countplot(x="Embarked", hue="Survived", data=titanic)
plt.figure()
sns.countplot(x="famsize", hue="Survived", data=titanic)

### Generate a Predictive Model ###
# Drop out unused columns for Machine Learning.
#titanic = titanic.drop(["Embarked", "Sex", "title",
#                        "Ticket", "CL", "Cabin", "Name"], axis=1)


