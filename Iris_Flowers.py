# importing Libraries 
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold # import KFold
from sklearn.metrics import confusion_matrix
import seaborn as sns

# import the Data .
dataset = pd.read_csv('data/Iris_Data.csv')
dataset.head()

# Test messing data . 
dataset.isna().sum()

# Splitting the Data to Features and Target .
X = dataset.iloc[:,:-1].values
y = dataset.iloc[: , -1].values

# Convert string data to numerical values .
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)


# Splitting the Data to Training and Test .
X_train , X_test , y_train , y_test = train_test_split(X ,
                                                       y ,
                                                       test_size = 0.3 ,
                                                       random_state = 0)

# Classification : 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

models = {'Logestic_Regression' : LogisticRegression() ,
          'KNN' : KNeighborsClassifier() ,
          'Random_Forest_Classifier' : RandomForestClassifier() ,
          'SVC' : SVC() ,
          'Decision_Tree' : DecisionTreeClassifier()
          }

def fit_and_score(models , X_train , X_test , y_train , y_test) :
    model_scores = {}
    model_confusion = {}
    for name , model in models.items() :
        # fitting the data :
        model.fit(X_train , y_train)
        model_scores[name] = model.score(X_test , y_test)
        y_predict = model.predict(X_test)
        model_confusion[name] = confusion_matrix(y_test , y_predict)
    return model_scores , model_confusion

fit_and_score(models = models ,
              X_train = X_train,X_test = X_test,
              y_train = y_train,y_test = y_test )

# All Algorithms have the same score .

# Visualization The Data .

sns.set_theme(style="ticks")
ax = sns.pairplot(dataset , hue = "species")
sns.barplot(data = dataset , x = dataset['species'] , y = dataset['sepal_length'])
sns.barplot(data = dataset , x = dataset['species'] , y = dataset['sepal_width'])
sns.barplot(data = dataset , x = dataset['species'] , y = dataset['petal_length'])
sns.barplot(data = dataset , x = dataset['species'] , y = dataset['petal_width'])

