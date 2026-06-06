from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StringType, DoubleType, MapType
from config.settings import KAFKA_BROKER, TOPIC_PRICES

spark = SparkSession.builder \
    .appName("PortfolioRiskStream") \
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

schema = StructType() \
    .add("timestamp", DoubleType()) \
    .add("prices", MapType(StringType(), DoubleType()))

def process_batch(df, epoch_id):
    for row in df.collect():
        print(f"[Epoch {epoch_id}] prices: {row['prices']}")

df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BROKER) \
    .option("subscribe", TOPIC_PRICES) \
    .load()

df_parsed = df_raw.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

query = df_parsed.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("append") \
    .start()

query.awaitTermination()