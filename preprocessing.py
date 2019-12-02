# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 21:46:06 2019

@author: dariy
"""
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler


# Create a SparkSession (the config bit is only for Windows!)
spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName("FraudDetection").getOrCreate()


dftrain = spark.read.format("csv").options(header='true', inferschema='true', delimiter=',').load('train_transaction.csv')
dftest = spark.read.format("csv").options(header='true', inferschema='true', delimiter=',').load('test_transaction.csv')
dftest = dftest.withColumn('isFraud', lit(2) )
dftest = dftest.select(dftrain.columns)
assert len(dftrain.columns) == len(dftest.columns)

dfall = dftrain.union( dftest.select(dftrain.columns) )

dfall.select('card3').distinct().rdd.map(lambda r: r[0]).collect()

catcol = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
          'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
          'M4', 'M1', 'M2',   'M3',  'M5', 'M6', 'M7', 'M8', 'M9',
]

for col in catcol:
    dfall = dfall.withColumn(col, dfall[col].cast('string')).fillna('uu', [col,])
dfall = dfall.fillna(-1, list(set(dfall.columns) - set(catcol)) )
    
stages = [StringIndexer(inputCol=col, outputCol=col+"Idx").setHandleInvalid("skip") 
            for col in catcol ]

pipeline = Pipeline(stages=stages)
df_idx = pipeline.fit(dfall).transform(dfall)

df_idx.select(sorted( catcol + [col+'Idx' for col in catcol] )).show()


encoder = OneHotEncoderEstimator(inputCols=[col+'Idx' for col in catcol],
                                 outputCols=[col+'Enc' for col in catcol])
df_enc = encoder.fit(df_idx).transform(df_idx)

df_enc.select(sorted( catcol + [col+'Enc' for col in catcol] )).show()

assembler = VectorAssembler(
    inputCols=[c for c in df_enc.columns 
               if c not in catcol+['TransactionID', 'isFraud', 'TransactionDT',]],
    outputCol="features")
output = assembler.transform(df_enc)
testDataset = output.select('TransactionID', 'features', 'isFraud').filter("isFraud = 2")
trainDataset = output.select('TransactionID', 'features', 'isFraud').filter("isFraud <> 2")


#%%
#from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.ml.classification import GBTClassifier

(trainingData, testData) = trainDataset.randomSplit([0.7, 0.3])

gbt = GBTClassifier(labelCol="isFraud", featuresCol="features", maxIter=10)
model = gbt.fit(trainingData)


from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'isFraud', maxIter=10)
lrModel = lr.fit(trainingData)

predictions = lrModel.transform(testData)

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol='isFraud')
print('Test Area Under ROC', evaluator.evaluate(predictions))
