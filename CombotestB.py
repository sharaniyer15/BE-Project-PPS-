import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier  as RFC
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as AS
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer

#import csv

#df = df.convert_objects(convert_numeric=True)



data = pd.read_csv("F:\DTC\SSC-HSC_CE-IT_11-12.csv", sep = ",")
#print(data.shape)
X =  data.iloc[:,1:9]
#print(X.dtypes)
y = data.iloc[:,9]
#print(X.shape)
#print(X.head(5))
#print(Y.head(5))
Y = y.to_frame()        
#print(type(Y))
scaler = StandardScaler().fit(X)
rescaledX=[]
rescaledX = scaler.transform(X)
np.set_printoptions(precision=3)
#print(rescaledX)
#warnings.filterwarnings(action='once')


#print(rescaledX.shape)
#print((type(rescaledX)))
df2= pd.DataFrame(rescaledX, columns=list('ABCDEFGH'))
#df2=pd.merge(df2,Y,right_index=True,left_index=True)
#df2['rescaledX'] = rescaledX
#print(df2)
print(df2.shape)
#print(Y.shape)

#2nd dataset

data1 = pd.read_csv("F:\DTC\SSC-HSC_CE-IT_12-13.csv", sep = ",")
X1 = data1.iloc[:,1:9]
y1 = data1.iloc[:,9]
Y1 = y1.to_frame()
scaler = StandardScaler().fit(X1)
rescaledY=[]
rescaledY = scaler.transform(X1)
df3= pd.DataFrame(rescaledY,columns=list('ABCDEFGH'))
#df3=pd.merge(df3,Y,right_index=True,left_index=True)
#print(rescaledY)
#df3['rescaledY'] = rescaledY.tolist()
#print(df3)
print(df3.shape)
#print(Y1.shape)
#3rd dataset

data2 = pd.read_csv("F:\DTC\SSC-HSC_CE-IT_13-14.csv", sep = ",")
X2 = data2.iloc[:,1:9]
y2 = data2.iloc[:,9]
Y2 = y2.to_frame()
scaler = StandardScaler().fit(X2)
rescaledZ=[]
rescaledZ = scaler.transform(X2)
df4= pd.DataFrame(rescaledZ, columns=list('ABCDEFGH'))
#df4=pd.merge(df4,Y,right_index=True,left_index=True)
#df4['rescaledZ'] = rescaledZ.tolist()
#print(df4)
print(df4.shape)
#print(Y2.shape)
df2 =df2.append(df3,ignore_index=True)

#All features merged in df4 as a single data frame
df4 = df4.append(df2,ignore_index=True)   
print(df4.shape)

Y = Y.append(Y1,ignore_index=True)
#print(Y)

#All Target values stored in Y2 as a single Data frame
Y2=Y2.append(Y,ignore_index=True)
#print(Y2)

columns_new = ['A','B','C','D','E','F','G','H']


""" Binning """

est = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='uniform')
est.fit(df4)
Xt = est.transform(df4)
Xt=pd.DataFrame(Xt, columns=columns_new)
#print(type(Xt))

# Training and testing
#Accuracy  = []
#for_depth = []
#for_leaf  = []
#
#df = pd.DataFrame()
#df['Accuracy']  = Accuracy
#df['max_depth'] = for_depth
#df['max_leaf_nodes'] = for_leaf 
#df = df.sort_values("Accuracy", ascending = False).head(10)
##print(df)

data2 = pd.read_csv("F:\DTC\SSC-HSC_CE-IT_14-15.csv", sep = ",")
X3 = data2.iloc[:,1:9]
y3 = data2.iloc[:,9]
Y3 = y3.to_frame()
scaler = StandardScaler().fit(X3)
rescaled=[]
rescaled = scaler.transform(X3)
test= pd.DataFrame(rescaled, columns=list('ABCDEFGH'))
#print(test.shape)
#print(test)
#print(test)

""" Binning """
est.fit(X3)
Xt2 = est.transform(X3)
print(Xt2.shape)

Xt2=pd.DataFrame(Xt2, columns=columns_new)
#print(type(Xt2))

Accuracy  = []
for_depth = []
for_leaf  = []
##print(rescaledX[0:5,:])
#with open("F:\BE Project Divided Data\CBSE-CBSE_CE-IT_14-15.csv", 'r', encoding="utf-8") as csvfile:
#    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
#    for row in spamreader:
for max_depth in range (1,10):
      for max_leaf_nodes in range(2,11):
#        print("For i and j = %s,%s"%(i,j))
        X_Train=Xt
        Y_Train=Y2
#        Y_Train = np.asarray(Y_Train, dtype = float)
#        Y_Train= Y_Train.ravel()
#        print(type(Y_Train))
        clf_gini=DTC(criterion = 'entropy',max_depth = max_depth,max_leaf_nodes =max_leaf_nodes)
        clf_gini.fit(X_Train, Y_Train)
        X_Test=Xt2
        Y_Test=y3
        shape=X_Test.shape[0]
        for i in range(shape):
            #print(i)
            Y_pred=clf_gini.predict(X_Test.loc[[i]])
            Y_pred = int(Y_pred[0])
            pred = clf_gini.predict_proba(X_Test.loc[[i]])
            pred = pred[0]
            print(Y_pred, ' ', Y_Test[i], ' ', pred[Y_pred])  
##        print('Y Test : ', Y_Test)
##        print('Y Pred : ', Y_pred)
            Y_pred = clf_gini.predict(X_Test)
#            print(" Accuracy is : ", AS(Y_Test,Y_pred)*100, ' ',max_depth, ' ',max_leaf_nodes)
##            print('Y Pred : ', pred)
            Accuracy.append(AS(Y_Test,Y_pred)*100)
            for_depth.append(max_depth)
            for_leaf.append(max_leaf_nodes)

print(max(Accuracy))
df = pd.DataFrame()
df['Accuracy']  = Accuracy
df['max_depth'] = for_depth
df['max_leaf_nodes'] = for_leaf 
df = df.sort_values("Accuracy", ascending = False).head(10)
print(df)

#""" Taking top 10 values of max_depth and max_leaf_nodes and testing"""
#df1 = pd.DataFrame()
#Accuracy     = []
#Random_State = []
#for_depth = []
#for_leaf  = []
##print(df)
##print(df.shape)
#for i in range((df.shape[0])):
#       for random_state in range(10,200,5):
#        X_Train, Y_Train = df4, Y2
#        clf_gini=DTC(criterion = 'entropy', random_state = random_state,max_depth = df.iloc[i,1]+1,max_leaf_nodes = df.iloc[i,2]+2)
#        clf_gini.fit(X_Train, Y_Train)
#        X_Test=test
#        Y_Test=y3
#        Y_pred=clf_gini.predict(X_Test)
##        print('Y Test : ', Y_Test)
##        print('Y Pred : ', Y_pred)
#        print(" Accuracy is : ", AS(Y_Test,Y_pred)*100)
#        Accuracy.append(AS(Y_Test,Y_pred)*100)
#        Random_State.append(random_state)
#        for_depth.append(df.iloc[i,1]+1)
#        for_leaf.append(df.iloc[i,2]+2)
#df1['Accuracy'] = Accuracy
#df1['max_depth'] = for_depth
#df1['max_leaf_nodes'] = for_leaf
#df1['Random_state'] = Random_State
#df1 = df1.sort_values("Accuracy", ascending = False).head(10)
##print(df1)
#Final_Accuracy = df1["Accuracy"].mean()
#
#Standard_Deviation = df1["Accuracy"].std()
#print("Final Accuracy is = ",Final_Accuracy,"with Standard Deviation = ",Standard_Deviation)
#print(confusion_matrix(Y_Test,Y_pred))
#print(classification_report(Y_Test,Y_pred))
