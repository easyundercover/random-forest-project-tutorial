#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sklearn
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

#Import data
url = 'https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv'
df = pd.read_csv(url)

#Clean and split data
drop_cols = ['PassengerId','Cabin', 'Ticket', 'Name']
df.drop(drop_cols, axis = 1, inplace = True)

X = df.drop("Survived",axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df_train = pd.concat([X_train, y_train], axis=1)

#Fill missing values
X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_test['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_train['Fare'].fillna(X_train['Fare'].median(), inplace=True)
X_test['Fare'].fillna(X_train['Fare'].median(), inplace=True)

#Encode cat variables
X_train[['Sex','Embarked']]=X_train[['Sex','Embarked']].astype('category')
X_test[['Sex','Embarked']]=X_test[['Sex','Embarked']].astype('category')
X_train['Sex']=X_train['Sex'].cat.codes
X_train['Embarked']=X_train['Embarked'].cat.codes
X_test['Sex']=X_test['Sex'].cat.codes
X_test['Embarked']=X_test['Embarked'].cat.codes

#Fit a Decision Tree model as comparison
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

#Fit Random Forest
clf2 = RandomForestClassifier(n_estimators=100, random_state=1107)
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)
accuracy_score(y_test, y_pred2)

#Randomized search
rf_cl = RandomForestClassifier(random_state=1107)
rf_cl_cv = RandomizedSearchCV(estimator = rf_cl, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=1107, n_jobs = -1) 
rf_cl_cv.fit(X_train,y_train)
rf_cl_cv.best_params_
rf_ht = RandomForestClassifier(random_state=1107, n_estimators=50, min_samples_split=10, min_samples_leaf=1, max_depth=50, bootstrap=True)
rf_ht.fit(X_train, y_train)
y_pred_rf = rf_ht.predict(X_test)

#Save model
filename = '../models/final_model.sav'
pickle.dump(rf_ht, open(filename, 'wb'))

#Algorithm boosting
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
y_xgb_pred = xgb.predict(X_test)
accuracy_score(y_test, y_xgb_pred)

#Gridsearch
xgb_2 = XGBClassifier()
parameters = {
     "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
     }
grid_xgb = RandomizedSearchCV(xgb_2,
                    parameters, n_jobs=4,
                    scoring="neg_log_loss",
                    cv=3)
grid_xgb.fit(X_train, y_train)

xgb_2 = grid_xgb.best_estimator_
y_pred_xgb_2 = xgb_2.predict(X_test)
accuracy_score(y_test, y_pred_xgb_2)
