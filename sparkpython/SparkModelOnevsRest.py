# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 17:19:16 2017

@author: Akshaykumar.Kore
"""

from pyspark.context import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler,StringIndexer,OneHotEncoder
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression,OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
sc = SparkContext()

sqlContext=SQLContext(sc)

data = sc.textFile('C:/Users/akshaykumar.kore/Downloads/data/kddcup.corrected').map(lambda line : line.split(","))
print("before Conversion data")
data=data.toDF()
print("after conversion data")

#data=data.na.fill(0)

#types = [f.dataType for f in data.schema.fields]

#print(types)
#data.toPandas().to_csv('C:/Users/akshaykumar.kore/Downloads/data/kddcup1.csv')

categoricalColumns = ["_2", "_3", "_4"]
stages = [] # stages in our Pipeline
for categoricalCol in categoricalColumns:
  # Category Indexing with StringIndexer
  stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol+"Index")
  # OneHotEncoder to convert categorical variables into binary SparseVectors
  encoder = OneHotEncoder(inputCol=categoricalCol+"Index", outputCol=categoricalCol+"classVec")
  # Add stages.
  stages += [stringIndexer, encoder]

#convert label into numerical
label_stringIdx = StringIndexer(inputCol = "_42", outputCol = "label")
stages += [label_stringIdx]

string="_"
numericCols=[string+str(i) for i in range(1,42)]

numericCols = [e for e in numericCols if e not in ('_2', '_3','_4')]


#numericCols = ["_1", "_5", "_6", "_7", "_8","_9","_10","_11","_12", "_13"]

categorical=[]
numerical=[]

for col in numericCols:
    data = data.withColumn(col+"index", data[col].cast(DoubleType())).drop(col)

for col in numericCols:
    col=col+"index"
    numerical.append(col)        

for c in categoricalColumns:
    c=c+"Index"
    categorical.append(c)
    
assemblerInputs = categorical + numerical
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

stages+=[assembler]

print("before fitting data")

pipeline=Pipeline(stages=stages)

model=pipeline.fit(data)
print("After fitting data")

dataset=model.transform(data)

selectedCols=["label","features"]

dataset=dataset.select(selectedCols)
(trainingData,testData) = dataset.randomSplit([0.7,0.3], seed=100)





model=OneVsRest.read().load("C:/Users/akshaykumar.kore/Downloads/machine learning/Spark/model")
predictions=model.transform(testData)
predictions.show()

'''
predictions.toPandas().to_csv('C:/Users/akshaykumar.kore/Downloads/data/kddcupPredictions.csv')
#predictions.show()
'''
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

accuracy=evaluator.evaluate(predictions)

print(accuracy)

#dataset.show()

print("succesfully run")