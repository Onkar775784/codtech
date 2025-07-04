from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, desc

# Step 1: Create Spark session
spark = SparkSession.builder \
    .appName("Electronics Review Analysis") \
    .config("spark.hadoop.io.nativeio.NativeIO$Windows.access0", "false") \
    .getOrCreate()


# Optional: Reduce logging noise
spark.sparkContext.setLogLevel("ERROR")

# Step 2: Load dataset
df = spark.read.json("reviews_Electronics.json")



# Step 3: Basic data exploration
print("Schema:")
df.printSchema()

print("Total number of records:", df.count())
df.select("overall", "reviewText", "reviewerID", "asin").show(5)

# Step 4: Data cleaning - Remove rows with null ratings or review text
df_clean = df.filter((col("overall").isNotNull()) & (col("reviewText").isNotNull()))

# Step 5: Analysis - Most reviewed products
top_products = df_clean.groupBy("asin").agg(count("*").alias("review_count")) \
    .orderBy(desc("review_count")).limit(10)
print("Top 10 most reviewed products:")
top_products.show()

# Step 6: Analysis - Average rating per product
avg_ratings = df_clean.groupBy("asin").agg(avg("overall").alias("avg_rating")) \
    .orderBy(desc("avg_rating")).limit(10)
print("Top 10 highest-rated products:")
avg_ratings.show()

# Step 7: Analysis - Most active reviewers
top_reviewers = df_clean.groupBy("reviewerID").agg(count("*").alias("reviews")) \
    .orderBy(desc("reviews")).limit(10)
print("Top 10 most active reviewers:")
top_reviewers.show()

# Stop Spark session
spark.stop()
