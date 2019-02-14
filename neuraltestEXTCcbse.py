import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score as AS


data1 = pd.read_csv("F:/DTC/CBSE-CBSE_EXTC_11-12.csv", sep = ",")
X1 =  data1.iloc[:,1:9]
y1 = data1.iloc[:,9]
Y1 = y1.to_frame()  
scaler = StandardScaler().fit(X1)
rescaledX = []
rescaledX = scaler.transform(X1)
np.set_printoptions(precision = 3)
df1 = pd.DataFrame(rescaledX, columns = list('ABCDEFGH'))

data2 = pd.read_csv("F:/DTC/CBSE-CBSE_EXTC_12-13.csv", sep = ",")
X2 = data2.iloc[:,1:9]
y2 = data2.iloc[:,9]
Y2 = y2.to_frame()
scaler = StandardScaler().fit(X2)
rescaledY = []
rescaledY = scaler.transform(X2)
df2= pd.DataFrame(rescaledY,columns=list('ABCDEFGH'))

data3 = pd.read_csv("F:/DTC/CBSE-CBSE_EXTC_13-14.csv", sep = ",")
X3 = data3.iloc[:,1:9]
y3 = data3.iloc[:,9]
Y3 = y3.to_frame()
scaler = StandardScaler().fit(X3)
rescaledZ = []
rescaledZ = scaler.transform(X3)
df3= pd.DataFrame(rescaledZ, columns=list('ABCDEFGH'))

df1 =df1.append(df2,ignore_index=True)
df3 = df3.append(df1,ignore_index=True) 

Y1 = Y1.append(Y2,ignore_index=True)
Y3 = Y3.append(Y1,ignore_index=True)

columns_new = ['A','B','C','D','E','F','G','H']
columns_new1 = ['A','B','C']
#print(df3.shape)
'''Training'''

X_train = pd.DataFrame(df3, columns=columns_new)
YR = Y3
Y_train = np.ravel(YR)
'''Testing'''

data4 = pd.read_csv("F:/DTC/CBSE-CBSE_EXTC_14-15.csv", sep = ",")
X4 = data4.iloc[:,1:9]
y4 = data4.iloc[:,9]
Y4 = y4.to_frame()
scaler = StandardScaler().fit(X4)
rescaled=[]
rescaled = scaler.transform(X4)
X_Test = pd.DataFrame(rescaled, columns=list('ABCDEFGH'))
Y_Test = y4
#print(X4.shape)
shape=X_Test.shape[0]
'''MLP classifier'''

from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import ParameterGrid

Accuracy  = []
for_depth = []
for_leaf  = []

#p = Perceptron(random_state=42,
#              max_iter=10)
#p.fit(X, y)

mlpc = MLPClassifier(hidden_layer_sizes=(15,15),solver='lbfgs')
mlpc.fit(X_train, Y_train)
for i in range(shape):
            #print(i)
            Y_pred=mlpc.predict(X_Test.loc[[i]])
            Y_pred = int(Y_pred[0])
            pred = mlpc.predict_proba(X_Test.loc[[i]])
            pred = pred[0]
            print(Y_pred, ' ', Y_Test[i], ' ', pred[Y_pred]) 
#         mlpc_result = mlpc.predict(X_test)
            Y_pred=mlpc.predict(X_Test)
            print(" Accuracy is : ", AS(Y_Test,Y_pred)*100)
            Accuracy.append(AS(Y_Test,Y_pred)*100)
                     
print(max(Accuracy))
df = pd.DataFrame()
df['Accuracy']  = Accuracy            
conf_matrix = confusion_matrix(Y_Test, Y_pred)
#accuracy = accuracy_score(Y_Test, Y_pred)
print(conf_matrix)



#param_grid = {'a': [1, 9], 'b': [True, False]}

#scores_10 = cross_val_score(estimator = mlpc,X = X_train, y = Y_train, cv = 10)
#scores1_10 = cross_val_score(estimator = mlpc,X = X_test, y = Y_test, cv = 10)
#f1_12_1 = f1_score(Y_test, mlpc_result, average='macro')
#f1_12_2 = f1_score(Y_test, mlpc_result, average='weighted')

#clf = GridSearchCV(mlpc, param_grid, cv=3, scoring='accuracy')
#clf.fit(X_train,Y_train)
#print("Best parameters set found on development set:")
#print(clf.best_params_)

