# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 19:34:43 2017

@author: Akshaykumar.Kore
"""

from pyspark.mllib.tree import RandomForest

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.feature import StringIndexer

from pyspark.ml import Pipeline
# --- Point 1, 2 ---
# Load and parse the data file into an RDD of LabeledPoint.
sc = SparkContext()
sqlContext = SQLContext(sc)
IRIS_rdd = sc.textFile('C:/Users/akshaykumar.kore/Downloads/iris.csv').map(lambda line : line.split(","))

header = IRIS_rdd.first()
IRIS_rdd = IRIS_rdd.filter(lambda line : line != header)

IRIS_rdd=IRIS_rdd.toDF()
IRIS_rdd=IRIS_rdd.withColumnRenamed("_1", "Sepal_length").withColumnRenamed("_2", "Sepal_Width").withColumnRenamed("_3", "Petal_Length").withColumnRenamed("_4", "petal_Width").withColumnRenamed("_5", "label")


#scaler=StandardScaler(inputCol='Sepal_length', outputCol='column')

#model=scaler.fit(IRIS_rdd)

#model.transform(IRIS_rdd).show()
IRIS_rdd.show()

training_set,test_set= IRIS_rdd.randomSplit([0.7,0.3])

indexers = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in list(set(IRIS_rdd.columns)) ]


#vecAssembler = VectorAssembler(inputCols=['Sepal_length','Sepal_Width','Petal_Length','petal_Width','label'], outputCol="features")
classifier=NaiveBayes(smoothing=1.0, modelType="multinomial")

pipeline=Pipeline(stages= [indexers,classifier])


model=pipeline.fit(training_set)

training_predictions=model.transform(training_set)
test_prediction=model.transform(test_set)
'''
# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = IRIS_rdd.randomSplit([0.7, 0.3])

classifier = NaiveBayes(smoothing=1.0, modelType="multinomial")

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
predictions.show()
#labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
#testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
#print('Test Error = ' + str(testErr))
#print('Learned classification forest model:')
#print(model.toDebugString())
'''