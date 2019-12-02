# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:39:13 2019

@author: dariy
"""
import pandas as pd
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark import SparkConf
config = SparkConf().setAll([('spark.executor.memory', '12g'), 
                   ('spark.driver.memory','12g'), ('spark.driver.maxResultSize', '12g')])
spark = SparkSession.builder.config(conf=config).master("local[*]").appName('ml-bank').getOrCreate()

#import findspark
#findspark.init()
#from pyspark.sql import SparkSession
#from pyspark.sql.functions import lit
#import pandas as pd
#spark = SparkSession.builder.appName('ml-bank').getOrCreate()

trainTrans = spark.read.csv('train_transaction.csv', header = True, inferSchema = True)
trainIdent = spark.read.csv('train_identity.csv', header = True, inferSchema = True)
trainIdent = trainIdent.withColumnRenamed('TransactionID', 'TransactionIDi' )
trainTransIdent = trainTrans.join( trainIdent, trainTrans.TransactionID==trainIdent.TransactionIDi, how='left')
#trainTransIdent = trainTransIdent.drop('TransactionIDi')
#pd.DataFrame(trainTransIdent.take(5), columns=trainTransIdent.columns).transpose()
#trainTrans.printSchema()

testTrans = spark.read.csv('test_transaction.csv', header = True, inferSchema = True)
testIdent = spark.read.csv('test_identity.csv', header = True, inferSchema = True)
testIdent = testIdent.withColumnRenamed('TransactionID', 'TransactionIDi' )
testTransIdent = testTrans.join( testIdent, testTrans.TransactionID == testIdent.TransactionIDi, how='left')
testTransIdent = testTransIdent.withColumn('isFraud', lit(2) )
#testTransIdent = testTransIdent.drop('TransactionIDi')

#pd.DataFrame(testIdentity.take(5), columns=testIdentity.columns).transpose()

#testTrans.printSchema()
df = trainTransIdent.union( testTransIdent.select( trainTransIdent.columns ) )
df = df.drop('TransactionIDi').drop('TransactionDT')

#%%
#pd.DataFrame(df.take(5), columns=df.columns).transpose()
#
## Summary
#numeric_features = [t[0] for t in df.dtypes if t[1] in ['int', 'double'] ]
#df.select(numeric_features).describe().toPandas().transpose()
## Correlations
#numeric_data = df.select(numeric_features).toPandas()
#axs = pd.scatter_matrix(numeric_data, figsize=(8, 8));
#n = len(numeric_data.columns)
#for i in range(n):
#    v = axs[i, 0]
#    v.yaxis.label.set_rotation(0)
#    v.yaxis.label.set_ha('right')
#    v.set_yticks(())
#    h = axs[n-1, i]
#    h.xaxis.label.set_rotation(90)
#    h.set_xticks(())
    
#%% Preparing Datasets
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler

columns = df.columns
categoricalColumns = [
        #Trans
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        # Ident
        'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19',
        'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27',
        'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35',
        'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo',]

stages = []
for cc in categoricalColumns:
    df = df.withColumn(cc, df[cc].cast('string')).fillna('uu', [cc,])
    stringIndexer = StringIndexer(inputCol = cc, outputCol = cc + 'Index')
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[cc + "classVec"])
    stages += [stringIndexer, encoder]
    
numericCols = [c for c in columns 
               if c not in categoricalColumns + ['TransactionID', 'isFraud',] ] 
df = df.fillna(-1, numericCols ) #list(set(dfall.columns) - set(catcol))

assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]


#%% Pipeline
from pyspark.ml import Pipeline
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedCols = ['isFraud', 'features'] + [c for c in columns if c!='isFraud']
df = df.select( selectedCols )
df.printSchema()

#%% Spliting Dataset
testAll = df.filter('isFraud = 2')
trainAll = df.filter('isFraud <> 2')

notFraudCount = trainAll.filter('isFraud = 0').count()
isFraudCount = trainAll.filter('isFraud = 1').count()

nSample = 400000
trainDataset = trainAll.filter('isFraud = 1').sample(True, nSample/2/isFraudCount )
trainDataset = trainDataset.union( trainAll.filter('isFraud = 0').sample(False, nSample/2/notFraudCount) )
trainDataset = trainDataset.sample(True, 1.01)

train, test = trainDataset.randomSplit([0.7, 0.3])

pd.DataFrame(train.take(5), columns=train.columns).transpose()

#%%
testAll = df.filter('isFraud = 2')
trainAll = df.filter('isFraud <> 2')

notFraudCount = trainAll.filter('isFraud = 0').count()
isFraudCount = trainAll.filter('isFraud = 1').count()

nSample = 400000
trainDataset = trainAll.filter('isFraud = 1').sample(True, nSample/2/isFraudCount )
trainDataset = trainDataset.union( trainAll.filter('isFraud = 0').sample(False, nSample/2/notFraudCount) )
trainDataset = trainDataset.sample(True, 1.01)

trainDatasetOut = trainDataset.select('isFraud', 'features')\
    .rdd.map( lambda r: [ r.isFraud,] + r.features.toArray().tolist() ).toDF()
    
trainDatasetOut.write.option("header", "true").csv("sparktrainDataSet.csv")
#    .toDF().toPandas()
    
import numpy as np
dataset = np.array( trainDatasetOut.collect() )
X = dataset[:,1:]
Y = dataset[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

#%% logistic Regression
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'isFraud', maxIter=20)
lrModel = lr.fit(train)

#%%
import matplotlib.pyplot as plt
import numpy as np
beta = np.sort(lrModel.coefficients)
plt.plot(beta)
plt.ylabel('Beta Coefficients')
plt.show()

trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()

#%% Logistic Regresion prediction
predictions = lrModel.transform(test)
predictions.select('TransactionID', 'isFraud', 'rawPrediction', 'prediction', 'probability').show(20)

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol='isFraud')
print('Test Area Under ROC', evaluator.evaluate(predictions))

#%% Decision Tree
from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'isFraud', maxDepth = 3)
dtModel = dt.fit(train); Model = dtModel
predictions = dtModel.transform(test)
predictions.select('TransactionID', 'isFraud', 'rawPrediction', 'prediction', 'probability').show(20)

evaluator = BinaryClassificationEvaluator( labelCol='isFraud' )
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()

#%% Gradient-Boosted Tree Classifier
from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(maxIter=10, featuresCol = 'features', labelCol = 'isFraud')
gbtModel = gbt.fit(train)
predictions = gbtModel.transform(test)
predictions.select('TransactionID', 'isFraud', 'rawPrediction', 'prediction', 'probability').show(10)

evaluator = BinaryClassificationEvaluator( labelCol='isFraud' )
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

#%% Output file

predictions = lrModel.transform(testAll)
predictions.select('TransactionID', 'isFraud', 'rawPrediction', 'prediction', 'probability').show(100)
dfout = predictions.select('TransactionID', 'probability')\
    .rdd.map( lambda r: (r.TransactionID, float( r.probability.toArray()[1]) ) )\
    .toDF(['TransactionID', 'isFraud']).toPandas()
    #    .withColumnRenamed('probability', 'isFraud')\

    
dfout.to_csv('submission20190929.csv', index=False)
    
#    .write.option("header", "true").csv("submission20190929.csv")

