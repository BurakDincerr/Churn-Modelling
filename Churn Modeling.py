# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 11:30:05 2021

@author: DINCERR
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Customer Churn modelling
data =pd.read_csv("Churn_Modelling.csv")

data.info()
data.head(2)

dum=pd.get_dummies(data["Geography"],drop_first=False)
dum2=pd.get_dummies(data["Gender"],drop_first=False)
x=data.drop(["RowNumber","CustomerId","Surname","Exited","Geography","Gender",],axis=1)
y=data["Exited"].values.reshape(-1,1)

x=pd.concat([x,dum],axis=1)
x=pd.concat([x,dum2],axis=1)
data["Geography"]

x=x.drop(["Female"],axis=1)
x.info()
x.head(1) 

x=x.values


# DATA SPLIT
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# DATA SCALING
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)



# MODEL
from xgboost import XGBClassifier
xgb=XGBClassifier(eta=0.3,max_depth=2, silent=2)

xgb.fit(x_train,y_train)
y_pred=xgb.predict(x_test)


#   Evaluate the accuracy of a classification.
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


from sklearn.model_selection import cross_val_score
cvs=cross_val_score(estimator=xgb,X=x_train,y=y_train,cv=4)
cvs.mean()# 0.85275 -> XGBOOST score



params = {
    'max_depth': [2,3],
    'eta': [0.4,0.6],
    'silent': [1,3]
    
}



from sklearn.model_selection import GridSearchCV
gsv=GridSearchCV(estimator=xgb,param_grid=params,cv=10,n_jobs=-1)

grid_search=gsv.fit(x_train,y_train)


grid_search.best_score_
grid_search.best_params_