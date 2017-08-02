Important uRl

1. Github naive bayes: https://gist.github.com/mitallast/87f0d0c5a8e5447c1626

2.Github random forest: https://github.com/XD-DENG/Spark-ML-Intro/commit/53aee429fb73d8b6a83f84cbdcb7f7ef3c968076#diff-df03c63e10371773f092d93affbdec42

3.Github Min max scaler: https://github.com/apache/spark/blob/master/examples/src/main/python/ml/min_max_scaler_example.py

4.Cancer data logistic regression---- https://mapr.com/blog/predicting-breast-cancer-using-apache-spark-machine-learning-logistic-regression/

5.About worker driver executor in spark cluster---https://stackoverflow.com/questions/32621990/what-are-workers-executors-cores-in-spark-standalone-cluster

6. don't think theano have spark support yet, however there are at least 2 deep learning libraries that are good with that. 
The first one is mxnet, they have support for spark, R, python and C++. Another option is deeplearning4j, made in Java with direct access to Spark.






Transformer: A Transformer is an algorithm which can transform one DataFrame into another DataFrame. 
E.g., an ML model is a Transformer which transforms a DataFrame with features into a DataFrame with predictions.

Estimator: An Estimator is an algorithm which can be fit on a DataFrame to produce a Transformer. 
E.g. a learning algorithm is an Estimator which trains on a DataFrame and produces a model.

Pipeline: A Pipeline chains multiple Transformers and Estimators together to specify an ML workflow.





---------------------

***Transformation**Extraction**Selection***

*Extraction: Extracting features from “raw” data
	Word2Vec---
	
	CountVectorizer---Extracts a vocabulary from document collections and generates a CountVectorizerModel.

	TF-IDF--- Term frequency-inverse document frequency (TF-IDF) is a feature vectorization method widely used in text mining to reflect the 
	importance of a term to a document in the corpus.

*Transformation: Scaling, converting, or modifying features

	Stringindexer---A label indexer that maps a string column of labels to an ML column of label indices. If the input column is numeric, 
	we cast it to string and index the string values. The indices are in [0, numLabels), ordered by label frequencies. So the most frequent label gets index 0.


	One Hot Encoder--- One-hot encoding maps a column of label indices to a column of binary vectors, with at most a single one-value. 
	This encoding allows algorithms which expect continuous features, such as Logistic Regression, to use categorical features.


	Vector Assembler---A feature transformer that merges multiple columns into a vector column.


	Vector Indexer---Class for indexing categorical feature columns in a dataset of Vector.


	MinMaxScaler---Rescale each feature individually to a common range [min, max] linearly using column summary statistics, 
	which is also known as min-max normalization or Rescaling. 


	HashingTF---Maps a sequence of terms to their term frequencies using the hashing trick. Currently we use Austin Appleby’s MurmurHash 3 algorithm (MurmurHash3_x86_32)
	to calculate the hash code value for the term object.


	IndexToString---Symmetrically to StringIndexer, IndexToString maps a column of label indices back to a column containing the original labels as strings.
	Applying IndexToString with categoryIndex as the input column, originalCategory as the output column, we are able to retrieve our original labels 
	(they will be inferred from the columns’ metadata)


	Tokenizer----Tokenization is the process of taking text (such as a sentence) and breaking it into individual terms (usually words). 
	A simple Tokenizer class provides this functionality.
				OR
	A tokenizer that converts the input string to lowercase and then splits it by white spaces.


	Binarizer--- Binarize a column of continuous features given a threshold.

	Bucketizer---Maps a column of continuous features to a column of feature buckets.

	DCT---


	ElementwiseProduct---Outputs the Hadamard product (i.e., the element-wise product) of each input vector with a provided 
	“weight” vector.In other words, it scales each column of the dataset by a scalar multiplier.


	N-Gram---A feature transformer that converts the input array of strings into an array of n-grams. Null values in the input array are ignored. 
	It returns an array of n-grams where each n-gram is represented by a space-separated string of words


	Normalizer---Normalize a vector to have unit norm using the given p-norm.


	PolynomialExpansion---Perform feature expansion in a polynomial space. As said in wikipedia of Polynomial Expansion,“In mathematics, an expansion of 
	a product of sums expresses it as a sum of products by using the fact that multiplication distributes over addition”. Take a 2-variable feature vector as
	an example: (x, y), if we want to expand it with degree 2, then we get (x, x * x, y, x * y, y * y).

	StandardScaler---Standardizes features by removing the mean and scaling to unit variance using column summary statistics on the samples in the training set.
	The “unit std” is computed using the corrected sample standard deviation, which is computed as the square root of the unbiased sample variance.


	StopWordsRemover---A feature transformer that filters out stop words from input. 
	Note: null values from input array are preserved unless adding null to stopWords explicitly.



*Selection: Selecting a subset from a larger set of features

	VectorSlicer---This class takes a feature vector and outputs a new feature vector with a subarray of the original features.




