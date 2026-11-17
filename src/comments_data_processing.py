# -*- coding: utf-8 -*-
import json
import requests
from datetime import datetime
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, current_timestamp, trim
from pyspark.sql.types import StringType
from google.cloud import bigquery

# ==========================================
# 1. Export Data From Api
# ==========================================
spark = SparkSession.builder \
    .appName("CommentsAPItoSpark") \
    .getOrCreate()

url = 'https://dummyjson.com/comments'
response = requests.get(url)

if response.status_code == 200:
    json_data = response.json()
    comments_list = json_data["comments"]

    json_strings = [json.dumps(item) for item in comments_list]
    rdd = spark.sparkContext.parallelize(json_strings)
    raw_df = spark.read.json(rdd)
else:
    print('Error Connection')

raw_df.printSchema()
raw_df.show(5)

# ==========================================
# 2. Bronze Layer
# ==========================================
# Unpacking internal dictionaries
df_bronze = raw_df.select(
    F.col('id'),
    F.col('body'),
    F.col('postId'),
    F.col('likes'),
    F.col('user.fullName').alias('user_fullName'),
    F.col('user.id').alias('user_id'),
    F.col('user.username').alias('user_username')
)

# Add bronze_loaded_at timestamp
df_bronze = df_bronze.withColumn("bronze_loaded_at", current_timestamp())

# Add unique surrogate key (comment_id) for lineage and deduplication
df_bronze = df_bronze.withColumn(
    'comment_id',
    F.concat(F.col("id"), F.lit("_"), F.col("bronze_loaded_at").cast("string"))
)

project_id = "dummyjson-e2e-pipeline"
table_id_bronze = f"{project_id}.bronze.comments"
client = bigquery.Client(project=project_id)

new_pandas_df = df_bronze.toPandas()

try:
    query = f"SELECT * FROM `{table_id_bronze}`"
    old_pandas_df = client.query(query).to_dataframe()
    combined_df = pd.concat([old_pandas_df, new_pandas_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['comment_id'], keep='last')
    print("Successfully merged old bronze data with new data locally.")
except Exception as e:
    print(f"Bronze table does not exist yet or first run. Using new data only. Error: {e}")
    combined_df = new_pandas_df

# Ensure string datatype for keys to prevent PyArrow type mismatch
combined_df['comment_id'] = combined_df['comment_id'].astype(str)

job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
job = client.load_table_from_dataframe(combined_df, table_id_bronze, job_config=job_config)
job.result()
print(f"Successfully uploaded {len(combined_df)} rows to {table_id_bronze}.")

# ==========================================
# 3. Cleaning Stage & Silver Layer
# ==========================================
# Filter out null/invalid IDs
df_bronze = df_bronze.filter(
    (col('id').isNotNull()) & 
    (col('id') > 0) & 
    (col('postId').isNotNull()) & 
    (col('postId') > 0)
)

# Clean text columns (Trim & InitCap)
text_column_names = ['user_fullName', 'user_username']
for name in text_column_names:
    df_bronze = df_bronze.withColumn(
        name, trim(F.initcap(F.regexp_replace(F.col(name), r"\s+", " ")))
    )

df_bronze = df_bronze.withColumn(
    'body', trim(F.regexp_replace(F.col('body'), r"\s+", " "))
)

# Add engagement_tier based on likes count
df_silver = df_bronze.withColumn(
    'engagement_tier',
    F.when(F.col('likes') >= 8, F.lit('High Engagement'))
     .when(F.col('likes') >= 3, F.lit('Moderate'))
     .otherwise(F.lit('Low'))
)

# Add silver_loaded_at column
df_silver = df_silver.withColumn("silver_loaded_at", current_timestamp())
table_id_silver = f"{project_id}.silver.comments"

new_pandas_df_silver = df_silver.toPandas()

try:
    query = f"SELECT * FROM `{table_id_silver}`"
    old_pandas_df = client.query(query).to_dataframe()
    combined_silver = pd.concat([old_pandas_df, new_pandas_df_silver], ignore_index=True)
    combined_silver = combined_silver.drop_duplicates(subset=['comment_id'], keep='last')
    print("Successfully merged old Silver data with new data locally.")
except Exception as e:
    print(f"Silver table does not exist yet or first run. Using new data only. Error: {e}")
    combined_silver = new_pandas_df_silver

combined_silver['comment_id'] = combined_silver['comment_id'].astype(str)

job = client.load_table_from_dataframe(combined_silver, table_id_silver, job_config=job_config)
job.result()
print(f"Successfully uploaded {len(combined_silver)} rows to {table_id_silver}.")

# ==========================================
# 4. Gold Layer (Star Schema Model)
# ==========================================
df_gold = df_silver.withColumn('gold_loaded_at', current_timestamp())

# Fact Comments Table
dim_comments = df_gold.select([
    'comment_id',
    'id',
    'postId',
    'likes',
    'engagement_tier',
    'gold_loaded_at',
    'silver_loaded_at',
    'bronze_loaded_at'
])

# Dimension Users Table
dim_comments_users = df_gold.select([
    'comment_id',
    'user_id',
    'user_fullName',
    'user_username',
    'gold_loaded_at',
    'silver_loaded_at',
    'bronze_loaded_at'
]).distinct()

gold_tables = {
    'dim_comments': dim_comments,
    'dim_comments_users': dim_comments_users
}

for table_name, df_table in gold_tables.items():
    table_id_gold = f"{project_id}.gold.{table_name}"
    new_pandas_df = df_table.toPandas()
    
    try:
        query = f"SELECT * FROM `{table_id_gold}`"
        old_pandas_df = client.query(query).to_dataframe()
        combined_gold = pd.concat([old_pandas_df, new_pandas_df], ignore_index=True)
        combined_gold = combined_gold.drop_duplicates(subset=['comment_id'], keep='last')
        print(f"Successfully merged old data for {table_name}.")
    except Exception as e:
        print(f"Table {table_name} does not exist yet. Using new data only. Error: {e}")
        combined_gold = new_pandas_df

    combined_gold['comment_id'] = combined_gold['comment_id'].astype(str)

    job = client.load_table_from_dataframe(combined_gold, table_id_gold, job_config=job_config)
    job.result()
    print(f"Successfully uploaded {len(combined_gold)} rows to {table_id_gold}.")
