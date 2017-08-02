# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 17:47:59 2017

@author: Akshaykumar.Kore
"""

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorIndexer
from pyspark import SparkSession

spark = SparkSession.builder.master("spark://50.50.50.226:7077").appName("adult1").getOrCreate()

df = spark.createDataFrame([(Vectors.dense([-1.0, 0.0]),),(Vectors.dense([0.0, 1.0]),), (Vectors.dense([0.0, 2.0]),)], ["a"])
indexer = VectorIndexer(maxCategories=2, inputCol="a", outputCol="indexed")
model = indexer.fit(df)
print(model.transform(df).head().indexed)
'''
model.numFeatures

model.categoryMaps

indexer.setParams(outputCol="test").fit(df).transform(df).collect()[1].test

params = {indexer.maxCategories: 3, indexer.outputCol: "vector"}
model2 = indexer.fit(df, params)
model2.transform(df).head().vector

vectorIndexerPath = temp_path + "/vector-indexer"
indexer.save(vectorIndexerPath)
loadedIndexer = VectorIndexer.load(vectorIndexerPath)
loadedIndexer.getMaxCategories() == indexer.getMaxCategories()

modelPath = temp_path + "/vector-indexer-model"
model.save(modelPath)
loadedModel = VectorIndexerModel.load(modelPath)
loadedModel.numFeatures == model.numFeatures

loadedModel.categoryMaps == model.categoryMaps
'''
print(successful)