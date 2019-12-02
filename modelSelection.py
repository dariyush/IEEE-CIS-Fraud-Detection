# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 23:05:53 2019

@author: dariy
"""


#%%
from tqdm import tqdm
from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBClassifier

#%%
#test = read_csv('test_transaction.csv')
#test['isFraud'] = 'test'
data = read_csv('train_transaction.csv')
#data = data.append(test, sort=False).reset_index(drop=True)


#%%
catcol = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
          'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
          'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',]
#df = data.drop(catcol+['TransactionDT'], axis=1)


df, le, oe = None, {}, {}
outMem = {}
for col in tqdm( data.columns ):
    if col in ['card1', 'card2', 'TransactionDT',]:
        continue
#    try:
    if col in catcol:
        label_encoder = LabelEncoder()
        feature = label_encoder.fit_transform( data[col].astype('str'))
        feature = feature.reshape(data.shape[0], 1)
        onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        feature = onehot_encoder.fit_transform(feature)
        fcols = [f'{col}_{i}' for i in range(feature.shape[1])]
        df1 = DataFrame(data=feature, columns=fcols)
        le[col] = label_encoder
        oe[col] = onehot_encoder
    else:
        df1 = data[col]

    if df is None:
        df = df1
    else:
        df = concat([df, df1], axis=1)
#    except Exception as e:
#        outMem[col] = len(set(data[col]))
#        print(e, col, len(set(data[col])) )
        

#%%
isFraud = df[ df['isFraud']==1 ]
lenIsFraud = len(isFraud) * 10
dataset = df[ df['isFraud']!=1 ].sample(frac=1)[:lenIsFraud]

while len(dataset) < 2*lenIsFraud:
    dataset = concat([dataset, isFraud])

dataset = dataset.values   
# split data into X and y
X = dataset[:,2:]
Y = dataset[:,1]

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)


#%%
# fit model no training data
model = XGBClassifier(max_depth=10, n_estimators=100, n_jobs=5)

model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred[:, 1], pos_label=1)
print( metrics.auc(fpr, tpr) )

#%%
test = read_csv('test_transaction.csv')
#dft = test.drop(catcol+['TransactionDT'], axis=1)

dft = None
for col in tqdm( test.columns ):
    if col in ['card1', 'card2', 'TransactionDT']:
        continue
    
    if col in catcol:
        feature = le[col].fit_transform( test[col].astype('str'))
        feature = feature.reshape(test.shape[0], 1)
        feature = oe[col].fit_transform(feature)
        fcols = [f'{col}_{i}' for i in range(feature.shape[1])]
        df1 = DataFrame(data=feature, columns=fcols)
    else:
        df1 = test[col]

    if dft is None:
        dft = df1
    else:
        dft = concat([dft, df1], axis=1)        

#%%
dfout = DataFrame()
dfout['TransactionID'] = dft['TransactionID']
Xt = dft.values[:, 1:] 
Yt = model.predict_proba( Xt )[:, 1]
dfout['isFraud'] = Yt
dfout.to_csv('submission20190923.csv', index=False)
