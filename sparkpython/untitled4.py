# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 18:43:12 2017

@author: Akshaykumar.Kore
"""

from pyspark import SparkContext
from pyspark.sql import SQLContext


sc=SparkContext(appName='Akkidemo')
sqlContext = SQLContext(sc)
IRIS_rdd = sc.textFile("C:/Users/akshaykumar.kore/Downloads/iris.csv").map(lambda line: line.split(","))

header = IRIS_rdd.first()
IRIS_rdd = IRIS_rdd.filter(lambda line : line != header)

IRIS_rdd=IRIS_rdd.toDF()
IRIS_rdd=IRIS_rdd.withColumnRenamed("_1", "Sepal_length").withColumnRenamed("_2", "Sepal_Width").withColumnRenamed("_3", "Petal_Length").withColumnRenamed("_4", "petal_Width").withColumnRenamed("_5", "label")


IRIS_rdd.show()