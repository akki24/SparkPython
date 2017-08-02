# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:57:30 2017

@author: Akshaykumar.Kore
"""


from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import StringIndexer,HashingTF

from pyspark.sql.types import DoubleType


#Setup
sc = SparkContext()
sqlContext = SQLContext(sc)

# Load and parse the data file into an RDD of LabeledPoint.
data = sc.textFile('C:/Users/akshaykumar.kore/Downloads/data/breast-cancer.txt').map(lambda line : line.split(","))
data=data.toDF()

data=data.drop(data._1)

names=data.schema.names

for name in names:
    data = data.withColumn(name+"index", data[name].cast(DoubleType())).drop(name)

types = [f.dataType for f in data.schema.fields]

print(types)

labelindexer= StringIndexer(inputCol="_11index",outputCol="label")

vect
pipeline=  Pipeline(stages=[labelindexer])
model=pipeline.fit(data).transform(data)

model.show()
