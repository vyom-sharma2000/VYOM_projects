# Databricks notebook source
# File location and type
file_location = "/FileStore/tables/Customer_Segmentation.csv"
file_type = "csv"

df = spark.read.csv(file_location, header= True, inferSchema = True)
df.show()

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('cluster').getOrCreate()
from pyspark.ml.feature import Imputer

imputer = Imputer(
    inputCols = ['Defaulted'],
    outputCols = ['Defaulted_imputed']).setStrategy('median')
df2  = imputer.fit(df).transform(df)
df2.show()
df2.na.drop()
df2.count()
df2.columns
df2.printSchema()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
Assembler = VectorAssembler(inputCols= ['Edu','Income','Years Employed','Card Debt','Other Debt','DebtIncomeRatio','Defaulted_imputed'],
                                outputCol = 'features')
output = Assembler.transform(df2)


# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
kmeans = KMeans(featuresCol = 'features')
model = kmeans.fit(output)
prediction = model.transform(output)
prediction.show()

# COMMAND ----------



# COMMAND ----------


