# Diabetes-prediction-BY-USING-KNN-ALGORITHM
THE PREDICTION OF DIABETES BY USING KNN ALGORITHM

--pandas
a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, real world data analysis in Python
train_test_split
Split arrays or matrices into random train and test subsets, providing two inputs it splits them to four.

KNeighborsClassifier
to solve classification problem using kNN algorithm
In [78]:




import pandas as pd
​
​
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix




read CSV file using pands funtion, then display its columns
In [79]:


dataset = pd.read_csv(r"C:\Users\ahmad shahid\Downloads\diabetes.csv")
print(len(dataset))
dataset.columns



768
Out[79]:
Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
      dtype='object')

features selection
predictor/independent and target/dependent features are selected using seperate variables for furture processing
In [80]:




features_selection = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
X = dataset[features_selection]
y = dataset.Outcome




splitting data
as data was selected using two variable X and y, now we'll have to split it so we can have seperate data form training and testing,following code is splitting X and y into X_train,X_test, y_train, y_test. here provided test size is 0.1 which means data will be splitted into 90:10 ratio. now let's have a look which variable will contain which part of data. X_train will have 90% independent values or 90% values of X variable, X_test will have 10% of indenpendent values. y_train will be containing 90% values of denpendent varibale y and y_test will contain 10% dependent values. 90% data will be used for training and 10% will be used for testing the trained model 
In [81]:


X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.1)




the number of nearest neighbors are selected wihch are 200 of them
In [82]:


classifier = KNeighborsClassifier(n_neighbors=200)




training
classifier/kNN algorithm is trained using training dataset
In [83]:


classifier.fit(X_train, y_train)


Out[83]:
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=213, p=2,
           weights='uniform')

prediction
In [84]:



array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int64)
       confusion_matrix(y_test,y_pred)

---
Out[90]:
array([[55,  0],
       [22,  0]], dtype=int64)
       
ACCURACY SCORE :
(TP+TN)/TOTAL
=(0+SS)/768
=0.0716145833
       
       
       
  2ND CASE :
  the number of nearest neighbors are selected wihch are 12 of them
In [127]:


classifier = KNeighborsClassifier(n_neighbors=12)




training
classifier/kNN algorithm is trained using training dataset
In [128]:


classifier.fit(X_train, y_train)


Out[128]:
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=12, p=2,
           weights='uniform')

prediction
In [129]:



y_pred = classifier.predict(X_test)
y_pred


Out[129]:
array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int64)
In [130]:


confusion_matrix(y_test,y_pred)


Out[130]:
array([[49,  5],
       [12, 11], dtype=int64)
ACCURACY SCORE :
(TP+TN)/TOTAL
(11+49)/768
=11.0638020833


----3RD CASE
the number of nearest neighbors are selected wihch are 20 of them
In [145]:


classifier = KNeighborsClassifier(n_neighbors=20)




training
classifier/kNN algorithm is trained using training dataset
In [146]:


classifier.fit(X_train, y_train)


Out[146]:
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=20, p=2,
           weights='uniform')

prediction
In [147]:



y_pred = classifier.predict(X_test)
y_pred


Out[147]:
array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
       1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], dtype=int64)

In [148]:


confusion_matrix(y_test,y_pred)


Out[148]:
array([[54,  2],
       [ 8, 13]], dtype=int64)
       ACCURACY SCORE :
       (TP+TN)/TOTAL
       (13+54)/768
       =54.0169270833
​




  
