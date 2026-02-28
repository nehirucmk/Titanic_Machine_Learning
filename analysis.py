import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# get dataset from internet
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
train_df = pd.read_csv(url)

# save the file 
train_df.to_csv('titanic_train.csv', index=False)

# show first 5 rows
print("data uploaded successfully")
print(train_df.head())

print("DATASET INFORMATION")
print(train_df.info())

print("MISSING VALUES")
print(train_df.isnull().sum())

print("DESCRIPTIVE STATISTICS")
print(train_df.describe())

# set the visual style
sns.set_theme(style="whitegrid")

# visualizing survival count by Gender
plt.figure(figsize=(6, 5))
sns.countplot(x='Survived', hue='Sex', data=train_df, palette='viridis')
plt.title('survival count by gender (0 = No, 1 = Yes)')
plt.xlabel('Survived')
plt.ylabel('Passenger Count')
plt.show()

# dropping columns (axis=1) that are not useful for prediction
train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# filling missing values in age with the median to keep the prediction stable
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

# filling missing 'Embarked' values with the mode (according to frequency)
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

print("missing values handled.")

# converting gender to numeric: male = 0, female = 1
train_df['Sex'] = train_df['Sex'].map({'male': 0,'female': 1})

# converting Embarked locations to numeric: S=0, C=1, Q=2
train_df['Embarked'] = train_df['Embarked'].map({'S': 0,'C': 1,'Q':2})

print("categorical data encoded.")

from sklearn.model_selection import train_test_split

X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"training set: {X_train.shape[0]} rows")
print(f"test set: {X_test.shape[0]} rows")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# train the model
model.fit(X_train, y_train)

# make predictions
predictions = model.predict(X_test)

# check how accurate the model is
accuracy = accuracy_score(y_test, predictions)
print(f"model accuracy: %{accuracy * 100:.2f}")

from sklearn.metrics import confusion_matrix, classification_report

# generating the confusion matrix
cm = confusion_matrix(y_test, predictions)
print("confusion matrix")
print(cm)

# detailed report
print("\nclassification report")
print(classification_report(y_test, predictions))

import numpy as np 

# what if i was in the ship?
# createting a sample passenger
# features: [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
# sex: 0 for male, 1 for female
# embarked: 0 for S, 1 for C, 2 for Q
my_data = np.array([[2, 0, 20, 0, 1, 15, 0]]) 

my_prediction = model.predict(my_data)

if my_prediction[0] == 1:
    print("\nresult: you survived!")
else:
    print("\nResult: you died :(")