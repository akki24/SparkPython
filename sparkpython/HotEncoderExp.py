# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 19:08:27 2017

@author: Akshaykumar.Kore
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 09:24:42 2017

@author: Akshaykumar.Kore
"""


'''
ONE HOT ENCODER--Maps a column of category indices to a column of binary vectors

'''

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

sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# Load and parse the data file into an RDD of LabeledPoint.
data = sc.textFile('C:/Users/akshaykumar.kore/Downloads/data/adults.csv').map(lambda line : line.split(","))
#data=sc.read.csv("C:/Users/akshaykumar.kore/Downloads/data/adult1.csv", header=True, mode="DROPMALFORMED")

data=data.toDF()

data=data.na.fill(0)

categoricalColumns = ["_2", "_4", "_6", "_7", "_8", "_9", "_10", "_14"]
stages = [] # stages in our Pipeline
for categoricalCol in categoricalColumns:
  # Category Indexing with StringIndexer
  stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol+"Index")
  # Use OneHotEncoder to convert categorical variables into binary SparseVectors
  encoder = OneHotEncoder(inputCol=categoricalCol+"Index", outputCol=categoricalCol+"classVec")
  # Add stages.  These are not run here, but will run all at once later on.
  stages += [stringIndexer, encoder]


pipeline=Pipeline(stages=stages)

model=pipeline.fit(data).transform(data)

model.show()

model.toPandas().to_csv("C:/Users/akshaykumar.kore/Downloads/data/adultHotEncoderExp.csv")
