# -*- coding: utf-8 -*-
import json
import requests
import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, current_timestamp, to_date, trim, lit, when
from pyspark.sql.types import StringType
from google.cloud import bigquery

# ==========================================
# 1. Export Data From API
# ==========================================
spark = SparkSession.builder \
    .appName("CartsAPItoSpark") \
    .getOrCreate()

url = 'https://dummyjson.com/carts'
response = requests.get(url)

if response.status_code == 200:
    json_data = response.json()
    carts_list = json_data["carts"]

    json_strings = [json.dumps(item) for item in carts_list]
    rdd = spark.sparkContext.parallelize(json_strings)
    raw_df = spark.read.json(rdd)
else:
    print('Error Connection')

raw_df.printSchema()

# ==========================================
# 2. Bronze Layer
# ==========================================
# Unpacking internal dictionaries and lists
df_bronze = raw_df.select(
    "id",
    "userId",
    "total",
    "discountedTotal",
    "totalProducts",
    "totalQuantity",
    F.explode("products").alias("product")
).select(
    F.col("id").alias("cart_id"),
    F.col("userId"),
    F.col("total"),
    F.col("discountedTotal"),
    F.col("totalProducts"),
    F.col("totalQuantity"),

    # Unpacking product columns
    F.col("product.id").alias("product_id"),
    F.col("product.title").alias("product_title"),
    F.col("product.price").alias("product_price"),
    F.col("product.quantity").alias("product_quantity"),
    F.col("product.total").alias("product_total_price"),
    F.col("product.discountPercentage").alias("product_discount_percentage"),
    F.col("product.discountedTotal").alias("product_discounted_total"),
    F.col("product.thumbnail").alias("product_thumbnail")
)

# إضافة وختم تاريخ البرونز (YYYY-MM-DD)
df_bronze = df_bronze.withColumn('bronze_loaded_at', current_timestamp())

# تكوين مفتاح فريد سليم للبند (Cart_ID + Product_ID + Date)
df_bronze = df_bronze.withColumn(
    'order_id',
    F.concat(
        F.col("cart_id").cast("string"),
        F.lit("_"),
        F.col("product_id").cast("string"),
        F.lit("_"),
        F.date_format(F.col("bronze_loaded_at"), "yyyy-MM-dd")
    )
)

project_id = "dummyjson-e2e-pipeline"
table_id_bronze = f"{project_id}.bronze.orders"
client = bigquery.Client(project=project_id)

new_pandas_df = df_bronze.toPandas()

try:
    query = f"SELECT * FROM `{table_id_bronze}`"
    old_pandas_df = client.query(query).to_dataframe()
    combined_df = pd.concat([old_pandas_df, new_pandas_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['order_id'], keep='last')
    print("Successfully merged old bronze data with new data locally.")
except Exception as e:
    print(f"Bronze table does not exist yet or first run. Using new data only. Error: {e}")
    combined_df = new_pandas_df

combined_df['order_id'] = combined_df['order_id'].astype(str)

job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
job = client.load_table_from_dataframe(combined_df, table_id_bronze, job_config=job_config)
job.result()
print(f"Successfully uploaded {len(combined_df)} rows to {table_id_bronze}.")

# ==========================================
# 3. Cleaning Stage & Silver Layer
# ==========================================
# تنظيف المسافات والروابط النصية
df_silver = df_bronze.withColumn('product_title', trim(F.regexp_replace(F.col('product_title'), r"\s+", " ")))
df_silver = df_silver.withColumn('product_title', F.initcap(F.col("product_title")))

df_silver = df_silver.withColumn('product_thumbnail', trim(F.regexp_replace(F.col('product_thumbnail'), r"\s+", " ")))
df_silver = df_silver.withColumn(
    "product_thumbnail",
    F.when(F.col("product_thumbnail").rlike("^https?://"), F.col("product_thumbnail"))
     .otherwise(F.lit("https://via.placeholder.com/150"))
)

# الفلترة الآمنة للمعرفات والأرقام
df_silver = df_silver.filter(
    (F.col("cart_id").isNotNull()) & (F.col("cart_id") > 0) &
    (F.col("userId").isNotNull()) & (F.col("userId") > 0) &
    (F.col("product_id").isNotNull()) & (F.col("product_id") > 0)
)

df_silver = df_silver.filter(
    (F.col('total') > 0) &
    ((F.col('discountedTotal') >= 0) | (F.col('discountedTotal').isNull())) &
    (F.col('totalProducts') > 0) &
    (F.col('totalQuantity') > 0) &
    (F.col('product_price') > 0) &
    (F.col('product_quantity') > 0) &
    (F.col('product_total_price') > 0) &
    (F.coalesce(F.col("product_discount_percentage"), F.lit(0.0)).between(0, 100)) &
    (F.col('product_discounted_total') >= 0)
)

# تقريب القيم الرقمية
numeric_cols = ['total', 'product_total_price', 'discountedTotal', 'product_price', 'product_quantity', 'product_discount_percentage', 'product_discounted_total']
for name in numeric_cols:
    df_silver = df_silver.withColumn(name, F.round(F.col(name), 2))

# إضافة عمود التدقيق للحسابات
df_silver = df_silver.withColumn(
    "calculation_audit_status",
    F.when(
        F.abs((F.col("product_price") * F.col("product_quantity")) - F.col("product_total_price")) >= 0.05,
        F.lit("ERROR: Product Total Price Mismatch")
    ).when(
        F.abs((F.col("product_total_price") * (1 - F.col("product_discount_percentage") / 100) - F.col("product_discounted_total"))) >= 0.05,
        F.lit("ERROR: Discounted Total Mismatch")
    ).otherwise(F.lit("ALL_VALID"))
)

# إضافة الخصومات والعملة قبل ترحيل السيلفر
df_silver = df_silver.withColumn("currency", lit("USD"))
df_silver = df_silver.withColumn(
    "product_discount_amount",
    F.round(F.col('product_total_price') - F.col('product_discounted_total'), 2)
)

# الختم الزمني لمرحلة السيلفر
df_silver = df_silver.withColumn("silver_loaded_at", current_timestamp())
table_id_silver = f"{project_id}.silver.orders"

new_pandas_df_silver = df_silver.toPandas()

try:
    query = f"SELECT * FROM `{table_id_silver}`"
    old_pandas_df = client.query(query).to_dataframe()
    combined_silver = pd.concat([old_pandas_df, new_pandas_df_silver], ignore_index=True)
    combined_silver = combined_silver.drop_duplicates(subset=['order_id'], keep='last')
    print("Successfully merged old Silver data with new data locally.")
except Exception as e:
    print(f"Silver table does not exist yet or first run. Using new data only. Error: {e}")
    combined_silver = new_pandas_df_silver

combined_silver['order_id'] = combined_silver['order_id'].astype(str)

job = client.load_table_from_dataframe(combined_silver, table_id_silver, job_config=job_config)
job.result()
print(f"Successfully uploaded {len(combined_silver)} rows to {table_id_silver}.")

# ==========================================
# 4. Gold Layer
# ==========================================
# إضافة ختم الجولد والتاريخ على البيانات المنظفة
df_gold = df_silver.withColumn("gold_loaded_at", current_timestamp())

fact_orders = df_gold.select(
    'cart_id',
    'order_id',
    F.col('userId').alias('user_id'),
    'product_id',
    'product_quantity',
    'product_price',
    'product_total_price',
    'product_discounted_total',
    'product_discount_amount',
    'bronze_loaded_at',
    'silver_loaded_at',
    'gold_loaded_at'
)

table_id_gold = f"{project_id}.gold.orders"
new_pandas_df_gold = fact_orders.toPandas()

try:
    query = f"SELECT * FROM `{table_id_gold}`"
    old_pandas_df = client.query(query).to_dataframe()
    combined_gold = pd.concat([old_pandas_df, new_pandas_df_gold], ignore_index=True)
    combined_gold = combined_gold.drop_duplicates(subset=['order_id'], keep='last')
    print("Successfully merged old Gold data with new data locally.")
except Exception as e:
    print(f"Gold table does not exist yet or first run. Using new data only. Error: {e}")
    combined_gold = new_pandas_df_gold

combined_gold['order_id'] = combined_gold['order_id'].astype(str)

job = client.load_table_from_dataframe(combined_gold, table_id_gold, job_config=job_config)
job.result()
print(f"Successfully uploaded {len(combined_gold)} rows to {table_id_gold}.")
