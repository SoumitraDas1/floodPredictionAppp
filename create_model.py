import numpy as np 
import pandas as pd 
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree 
import pickle

df = pd.read_csv("Sample_Data.csv")
print(df)
print(df.columns)

features = ['temperature', 'Humidity', 'Rain_No_of_Days', 'Measure_of_Rain']
X = df[features]
Y = df['Flooded']
print(Y)

#train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, Y, test_size=0.2, random_state = 0)

# print(train_X)
# print(val_X)

#Fit model: Capture patterns from provided data. This is the heart of modeling.

dtree = DecisionTreeClassifier()

dtree = dtree.fit(train_X,train_y)

#prediction
print(dtree.predict(X.head()))
print(Y.head())

# Manual prediction
input_data = np.array([[25, 80, 6, 70]])
print(dtree.predict(input_data))

# # Save the model to a file
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(dtree, file)

predict_flood = dtree.predict(X)
accuracy = accuracy_score(Y,predict_flood)
print(f"Accuracy Score: {accuracy}")
