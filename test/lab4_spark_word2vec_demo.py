import os
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, split
from pyspark.ml.feature import Word2Vec

os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["SPARK_LOCAL_DIRS"] = "C:\\spark-temp"
os.makedirs("C:\\hadoop\\bin", exist_ok=True)
os.makedirs("C:\\spark-temp", exist_ok=True)

spark = (
    SparkSession.builder.appName("Lab4_Spark_Word2Vec_C4")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .getOrCreate()
)

data_path = "data/c4-train.00000-of-01024-30K.json"

try:
    df = spark.read.json(data_path)
except Exception as e:
    print(f"Error reading JSON file: {e}")
    spark.stop()
    exit()

df_clean = (
    df.select("text")
    .withColumn("text", lower(col("text")))
    .withColumn("text", regexp_replace(col("text"), "[^a-zA-Z\\s]", ""))
    .withColumn("words", split(col("text"), "\\s+"))
    .filter(col("words").isNotNull())
)

word2Vec = Word2Vec(
    vectorSize=100,
    minCount=5,
    inputCol="words",
    outputCol="result"
)

model = word2Vec.fit(df_clean)

word = "computer"
try:
    synonyms = model.findSynonyms(word, 5)
    print(f"Top 5 synonyms for '{word}':")
    synonyms.show()
except Exception as e:
    print(f"Could not find synonyms for '{word}': {e}")

spark.stop()