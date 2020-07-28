# Random_Forest_Project
For this project I will be exploring publicly available data from LendingClub.com.
I use lending data from 2007-2010 and be trying to classify and predict whether or not the borrower paid back their loan in full.

Here are what the columns represent:

- credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
- purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
- int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
- installment: The monthly installments owed by the borrower if the loan is funded.
- log.annual.inc: The natural log of the self-reported annual income of the borrower.
- dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
- fico: The FICO credit score of the borrower.
- days.with.cr.line: The number of days the borrower has had a credit line.
- revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
- revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
- inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
- delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
- pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

## Importing Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

## Getting the Data

```python
loans = pd.read_csv('loan_data.csv')
loans.info()
```
<img src= "https://user-images.githubusercontent.com/66487971/88571713-bae3c980-d046-11ea-97b8-5d1d24bc6f74.png" width = 350>

## Exploratory Data Analysis

```python
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
```

<img src= "https://user-images.githubusercontent.com/66487971/88637938-777b7080-d0c3-11ea-8e70-ec3fb5096b0d.png" width = 600>

```python
plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')

```
<img src= "https://user-images.githubusercontent.com/66487971/88638048-a1cd2e00-d0c3-11ea-9ef4-873b031b22bc.png" width = 600>

```python
plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')
```

<img src= "https://user-images.githubusercontent.com/66487971/88638226-e0fb7f00-d0c3-11ea-8827-1af668d1950b.png" width = 600>

```python
sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')
```

<img src= "https://user-images.githubusercontent.com/66487971/88638318-04262e80-d0c4-11ea-8ffd-1eb84af99f59.png" width = 600>

```python
plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')
    
```

<img src= "https://user-images.githubusercontent.com/66487971/88638460-346dcd00-d0c4-11ea-8966-2fb7921e1203.png" width = 800>

# Setting up the Data

## Transforming dummy variables

```python
cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data.info()
```
<img src= "https://user-images.githubusercontent.com/66487971/88639295-4c921c00-d0c5-11ea-9ed7-21ba40e71e02.png" width = 400>

# Train Test Split

```python
from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

```

# Training a Decision Tree Model

```python

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

```

# Predictions and Evaluation of Decision Tree

```python

predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
```

<img src= "https://user-images.githubusercontent.com/66487971/88639793-d8a44380-d0c5-11ea-8d5c-0e0e2f1f5277.png" width = 400>

```python
print(confusion_matrix(y_test,predictions))
```
<img src= "https://user-images.githubusercontent.com/66487971/88639914-fbcef300-d0c5-11ea-9634-beda386c3da1.png" width = 300>


# Training the Random Forest model

```python
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
```















