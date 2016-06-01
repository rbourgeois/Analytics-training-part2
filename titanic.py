# -*- coding: utf-8 -*-
"""
Created on Sun May 29 11:49:53 2016

@author: Romain Bourgeois
Date : 29th May 2016
Revised : 29th May 2016
"""
import pylab as P ##histogram
import numpy as np
import pandas as pd
import csv as csv

from sklearn.ensemble import RandomForestClassifier
from collections import Counter #counter
from sklearn.tree import DecisionTreeClassifier

wd = 'C:/Users/aisromainbou/Documents/AXA/Python-training/Part2/'
trainFile = 'train.csv'
testFile = 'test.csv'
train_df = pd.read_csv(wd+trainFile, header=0)
test_df = pd.read_csv(wd+testFile, header=0)
test_df['Survived'] = 'nan'
combi_df = train_df.append(test_df) ##, ignore_index=True

## basic statistics
## check char variables: give number of occurrences for each element
Counter(combi_df['Sex']) ## no missing values
Counter(combi_df['Embarked']) ##2 missing values
Counter(combi_df['Cabin']) ## 1014 missing values
## check numeric variables
combi_df.describe() ##Age and Fare have missing values

## function for extracting honorific (i.e. title) from the Name feature
combi_df['Title'] = combi_df['Name'].str.split('[,|.]').str[1]
Counter(combi_df['Title'])
combi_df['Title'] = combi_df['Title'].str.replace(' ','')
## clean title
Title_orig = ['Mlle|Ms','Sira|Lady|Dona|theCountess|Jonkheer','Mme','Capt|Col|Rev|Dr|Don|Major|Sir']
Title_clean = ['Miss','Lady','Mrs','Sir']
for i in range(0,4):
    combi_df['Title'] = combi_df['Title'].str.replace(Title_orig[i],Title_clean[i])

## clean Embarked
combi_df.loc[(combi_df['Embarked'].isnull()),'Embarked'] = 'S'

## clean Fare
Fare_median3 = combi_df[combi_df.Pclass == 3 ]['Fare'].dropna().median() ##8.05
combi_df.loc[combi_df['Fare'].isnull(),'Fare'] = Fare_median3

## Group fare
combi_df['FareGroup']	= 'UNK'	
combi_df.loc[(combi_df['Fare'] <= 7.5), 'FareGroup'] = '1. 0-7.5'
combi_df.loc[(combi_df['Fare'] > 7.5) & (combi_df['Fare'] <= 15), 'FareGroup'] = '2. 7.5-15'
combi_df.loc[(combi_df['Fare'] > 15) & (combi_df['Fare'] <= 30), 'FareGroup'] = '3. 15-30'
combi_df.loc[(combi_df['Fare'] > 30), 'FareGroup'] = '4. 30+'
Counter(combi_df['FareGroup'])

## create family variable
combi_df['Family'] = combi_df['SibSp'] + combi_df['Parch'] + 1

## create Fare.pp attempts to adjust group purchases by size of family
combi_df['Fare.pp'] = combi_df['Fare'] / combi_df['Family']

## First character in Cabin number represents the Deck 
combi_df['Deck'] = combi_df['Cabin'].str[0]
combi_df.loc[(combi_df['Deck'].isnull()),'Deck'] = 'UNK'
combi_df.loc[(combi_df['Deck'] == 'T'),'Deck'] = 'UNK'
combi_df.loc[(combi_df['Deck'] == 'G'),'Deck'] = 'F'
## Odd-numbered cabins were reportedly on the port side of the ship
## Even-numbered cabins assigned Side="starboard"
combi_df['tempSide'] = combi_df['Cabin'].str[-1]
combi_df['Side'] = 'UNK'
combi_df.loc[(combi_df['tempSide'] == '0') |
             (combi_df['tempSide'] == '2') |
             (combi_df['tempSide'] == '4') |
             (combi_df['tempSide'] == '6') |
             (combi_df['tempSide'] == '8') ,'Side'] = 'port'
combi_df.loc[(combi_df['tempSide'] == '1') |
             (combi_df['tempSide'] == '3') |
             (combi_df['tempSide'] == '5') |
             (combi_df['tempSide'] == '7') |
             (combi_df['tempSide'] == '9') ,'Side'] = 'starboard'
combi_df = combi_df.drop('tempSide', 1)

## Create FamilyID variable
combi_df['Surname'] = combi_df['Name'].str.split('[,|.]').str[0]
combi_df['FamilyID'] = combi_df['Family'].map(str) + combi_df['Surname']
combi_df = combi_df.drop('Surname', 1)
combi_df.loc[(combi_df['Family'] == 1), 'FamilyID'] = '1'
combi_df['FamilyID2'] = combi_df['FamilyID']
combi_df.loc[(combi_df['Family'] == 2), 'FamilyID2'] = '2'
combi_df['FamilyID3'] = combi_df['FamilyID2']
combi_df.loc[(combi_df['Family'] == 3), 'FamilyID3'] = '3'

## Group age
combi_df['AgeGroup']	= 'UNK'	
combi_df.loc[(combi_df['Age'] <= 20), 'AgeGroup'] = '0-20'
combi_df.loc[(combi_df['Age'] > 20) & (combi_df['Age'] <= 30), 'AgeGroup'] = '20-30'
combi_df.loc[(combi_df['Age'] > 30) & (combi_df['Age'] <= 40), 'AgeGroup'] = '30-40'
combi_df.loc[(combi_df['Age'] > 40), 'AgeGroup'] = '40+'

## Replace age missing values by decision tree
combi_df['Child'] = 0
combi_df.loc[(combi_df['Title'] == 'Master') | (combi_df['Title'] == 'Miss') ,'Child'] = 1 
Agefit_df = combi_df.loc[(combi_df['Age'].notnull()),['Pclass','Parch','Fare','Child','Age']]
target_df=Agefit_df[['Age']].astype(int)
features = list(Agefit_df.columns[:4])
Agefit = DecisionTreeClassifier(random_state=0)
Agefit = Agefit.fit(Agefit_df[features],target_df)

## Display decision tree
Agenull_df = combi_df.loc[(combi_df['Age'].isnull()),['Pclass','Parch','Fare','Child']] 
Agenull_df['Age'] = Agefit.predict(Agenull_df)         
combi_df.loc[(combi_df['Age'].isnull()),'Age'] = Agenull_df['Age']  
## Group clean age 
combi_df['AgeGroupClean']	= 'UNK'	
combi_df.loc[(combi_df['Age'] <= 20), 'AgeGroupClean'] = '0-20'
combi_df.loc[(combi_df['Age'] > 20) & (combi_df['Age'] <= 30), 'AgeGroupClean'] = '20-30'
combi_df.loc[(combi_df['Age'] > 30) & (combi_df['Age'] <= 40), 'AgeGroupClean'] = '30-40'
combi_df.loc[(combi_df['Age'] > 40), 'AgeGroupClean'] = '40+'

## Create Child variable
combi_df['Child'] = 0
combi_df.loc[(combi_df['Age'] <= 12), 'Child'] = 1

## Create Mother variable
combi_df['Mother'] = 0
combi_df.loc[(combi_df['Title'] == 'Mrs') & (combi_df['Parch'] > 0), 'Mother'] = 1

## create a family member survived/died variable

## format final database
combi_df= combi_df.drop(['Name', 'Ticket', 'Cabin','FamilyID'], axis=1) 
combi_df['Sex'] = combi_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
combi_df['Embarked'] = combi_df['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)
combi_df['FareGroup'] = combi_df['FareGroup'].map( {'1. 0-7.5': 0, '2. 7.5-15': 1, '3. 15-30': 2, '4. 30+': 3} ).astype(int)
combi_df['Deck'] = combi_df['Deck'].map( {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'UNK': 6} ).astype(int)
combi_df['Side'] = combi_df['Side'].map( {'starboard': 0, 'port': 1, 'UNK': 2} ).astype(int)
combi_df['AgeGroup'] = combi_df['AgeGroup'].map( {'0-20': 0, '20-30': 1, '30-40': 2, '40+': 3, 'UNK': 4} ).astype(int)
combi_df['AgeGroupClean'] = combi_df['AgeGroupClean'].map( {'0-20': 0, '20-30': 1, '30-40': 2, '40+': 3} ).astype(int)
combi_df['Title'] = combi_df['Title'].map( {'Miss': 0, 'Master': 1, 'Mrs': 2, 'Mr': 3, 'Sir': 4, 'Lady': 5} ).astype(int)

# Collect the test data's PassengerIds before dropping it
ids = combi_df['PassengerId'].values
ids = ids[891:]
combi_df= combi_df.drop(['FamilyID2','FamilyID3', 'PassengerId'], axis=1) 
combi_df= combi_df[['Survived','Age','Embarked','Fare','Parch','Pclass',
'Sex','SibSp','Title','FareGroup','Family','Fare.pp','Deck','Side',
'AgeGroup','Child','AgeGroupClean','Mother']]

## start modelling!!
train_clean_df = combi_df[:891]
test_clean_df = combi_df[891:]
train_clean_df['Survived']=train_clean_df['Survived'].astype(int)
# Convert back to a numpy array
train_data = train_clean_df.values
test_data = test_clean_df.values
 
print 'Training...'
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting...'
output = forest.predict(test_data[0::,1::]).astype(int)


predictions_file = open("submit1rf.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'