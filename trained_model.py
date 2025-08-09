import pandas as pd
import streamlit as slt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from math import sqrt
import joblib

# Import data
df = pd.read_csv('mtcars.csv')

# Split data into train and test sets
X = df.drop(columns=['model', 'qsec'])
y = df['qsec']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train decision tree
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

joblib.dump(dt, 'decision_tree_model.pkl')
