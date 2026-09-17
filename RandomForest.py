# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def rf():
    # Importing the training dataset
    train_dataset = pd.read_csv('train_set.csv')
    X_train = train_dataset.iloc[:, [2,3,4,5]].values
    y_train = train_dataset.iloc[:, 6].values
    
    
    # Importing the testing dataset
    test_dataset = pd.read_csv('test.csv')
    X_test = test_dataset.iloc[:, [2,3,4,5]].values
    y_test = test_dataset.iloc[:, 6].values
    
    
    
    '''
    labelencoder_X = LabelEncoder()
    #X_train[:, 2] = labelencoder_X.fit_transform(X_train[:, 2])
    #onehotencoder = OneHotEncoder(categorical_features = [2])
    #X_train = onehotencoder.fit_transform(X_train).toarray()
    
    X_train[:, 3] = labelencoder_X.fit_transform(X_train[:, 3])
    onehotencoder = OneHotEncoder(categorical_features = [3])
    X_train = onehotencoder.fit_transform(X_train).toarray()
    
    X_train[:, 4] = labelencoder_X.fit_transform(X_train[:, 4])
    onehotencoder = OneHotEncoder(categorical_features = [4])
    X_train = onehotencoder.fit_transform(X_train).toarray()
    
    X_train[:, 5] = labelencoder_X.fit_transform(X_train[:, 5])
    onehotencoder = OneHotEncoder(categorical_features = [5])
    X_train = onehotencoder.fit_transform(X_train).toarray()
    
    
    # Encoding categorical data
    # Encoding the Independent Variable
    labelencoder_XY = LabelEncoder()
    X_test[:, 2] = labelencoder_XY.fit_transform(X_test[:, 2])
    onehotencoder = OneHotEncoder(categorical_features = [2])
    X_test = onehotencoder.fit_transform(X_test).toarray()
    
    X_test[:, 3] = labelencoder_XY.fit_transform(X_test[:, 3])
    onehotencoder = OneHotEncoder(categorical_features = [3])
    X_test = onehotencoder.fit_transform(X_test).toarray()
    
    X_test[:, 4] = labelencoder_XY.fit_transform(X_test[:, 4])
    onehotencoder = OneHotEncoder(categorical_features = [4])
    X_test = onehotencoder.fit_transform(X_test).toarray()
    
    X_test[:, 5] = labelencoder_XY.fit_transform(X_test[:, 5])
    onehotencoder = OneHotEncoder(categorical_features = [5])
    X_test = onehotencoder.fit_transform(X_test).toarray()
    '''
    
    
    
    # Feature Scaling
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
    
    
    # Fitting Random Forest Classification to the Training set
    
    rand_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
    rand_classifier.fit(X_train, y_train)
    
    
    # Predicting the Test set results
    y_pred = rand_classifier.predict(X_test)
    print('Predicted output for RF')
    print(y_pred)
    
    
    # Making the Confusion Matrix
    
    ac1 = accuracy_score(y_test, y_pred)
    print('Accuracy for RF')
    print(ac1)
    
    # Making the Confusion Matrix
    
    cm1 = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix for RF')
    print(cm1)
    
    
    fs = f1_score(y_test, y_pred, average='micro') 
    print('F1 score for RF')
    print(fs)










