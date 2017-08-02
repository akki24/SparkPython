# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:31:56 2017

@author: Akshaykumar.Kore
"""

from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SQLContext,SparkSession
from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
#Setup
#spark = SparkSession.builder.master("spark://master:7077").appName("adult").config("spark.some.config.option", "akki").getOrCreate().enableHiveSupport()



## Initialize SparkContext. 
sc = SparkContext()
sqlContext=SQLContext(sc)
# Load and parse the data file into an RDD of LabeledPoint.
data = sc.textFile('C:/Users/akshaykumar.kore/Downloads/data/adults.csv').map(lambda line : line.split(","))
#data=sc.read.csv("C:/Users/akshaykumar.kore/Downloads/data/adult1.csv", header=True, mode="DROPMALFORMED")

data=data.toDF()
data.show()
data=data.na.fill(0)

categoricalColumns = ["_2", "_4", "_6", "_7", "_8", "_9", "_10", "_14"]
stages = [] # stages in our Pipeline
for categoricalCol in categoricalColumns:
  # Category Indexing with StringIndexer
  stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol+"Index")
  # OneHotEncoder to convert categorical variables into binary SparseVectors
 # encoder = OneHotEncoder(inputCol=categoricalCol+"Index", outputCol=categoricalCol+"classVec")
  # Add stages.
  stages += [stringIndexer]#, encoder]

#convert label into numerical
label_stringIdx = StringIndexer(inputCol = "_15", outputCol = "label")
stages += [label_stringIdx]
numericCols = ["_1", "_3", "_5", "_11", "_12", "_13"]

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

#fitting all the stages in pipeline
pipeline = Pipeline(stages=stages)
data = pipeline.fit(data).transform(data)


selectedCols=["label","features"]

datasets=data.select(selectedCols)

#datasets.show()

(trainingData,testData) = datasets.randomSplit([0.7,0.3], seed=100)

#applying machine learning model
classifier=LogisticRegression(maxIter=2,labelCol="label",featuresCol="features")

#Evaluator for evaluating
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")

grid=ParamGridBuilder().addGrid(classifier.regParam,[1.0,2.0]).addGrid(classifier.maxIter,[10,15]).build()

cv = CrossValidator(estimator=classifier, estimatorParamMaps=grid, evaluator=evaluator)

model=cv.fit(trainingData)

predictions=model.transform(testData)

predictions.show()

print(evaluator.evaluate(predictions))

'''
model=classifier.fit(trainingData)

#print(model.coefficients)
predictions=model.transform(testData)

#predictions.show()
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")

metric = evaluator.evaluate(predictions)

#accuracy by "receiver operating characteristic" curve
print ("areaUnderROC metric is =%g "  % metric)

metric = evaluator.evaluate(predictions,{ evaluator.metricName : "areaUnderPR"})
#accuracy by "precision and recall" curve
print ("areaUnderPR metric is =%g "  % metric)
'''
print("successfully running")