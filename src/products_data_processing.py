# -*- coding: utf-8 -*-
import json
import requests
from datetime import datetime
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, concat_ws, current_timestamp
from pyspark.sql.types import StringType
from google.cloud import bigquery

# ==========================================
# 1. Export Data From Api
# ==========================================
spark = SparkSession.builder \
    .appName("RestaurantProductsAPItoSpark") \
    .getOrCreate()

url = 'https://dummyjson.com/products'
response = requests.get(url)

if response.status_code == 200:
    json_data = response.json()
    products_list = json_data["products"]

    json_strings = [json.dumps(item) for item in products_list]
    rdd = spark.sparkContext.parallelize(json_strings)
    raw_df = spark.read.json(rdd)
else:
    print('Error Connection')

raw_df.printSchema()
raw_df.show()

# ==========================================
# 2. Bronze Layer
# ==========================================
df_bronze = raw_df.select(
        col("id"),
        col("title"),
        col("description"),
        col("category"),
        col("price"),
        col("discountPercentage"),
        col("rating"),
        col("stock"),
        col("brand"),
        col("sku"),
        col("weight"),
        concat_ws(',', col('tags')).alias('all_tags'),
        col("meta.createdAt").alias("created_at"),
        col("meta.updatedAt").alias("updated_at"),
        col("meta.barcode").alias("barcode"),
        col("dimensions.width").alias("dimension_width"),
        col("dimensions.height").alias("dimension_height"),
        col("dimensions.depth").alias("dimension_depth"),
        col("warrantyInformation"),
        col("shippingInformation"),
        col("availabilityStatus"),
        col("returnPolicy"),
        col("minimumOrderQuantity")
)

df_bronze = df_bronze.withColumn("bronze_loaded_at", current_timestamp())
df_bronze = df_bronze.withColumn('product_id',
             F.concat(F.col("id"),
                      F.lit("_"),
                      F.col("bronze_loaded_at").cast("string")
                      )
            )

project_id = "dummyjson-e2e-pipeline"
table_id_bronze = f"{project_id}.bronze.products"
client = bigquery.Client(project=project_id)

new_pandas_df = df_bronze.toPandas()

try:
    query = f"SELECT * FROM `{table_id_bronze}`"
    old_pandas_df = client.query(query).to_dataframe()
    combined_df = pd.concat([old_pandas_df, new_pandas_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['product_id'], keep='last')
    print("Successfully merged old bronze data with new data locally.")
except Exception as e:
    print(f"Bronze table does not exist yet or first run. Using new data only. Error: {e}")
    combined_df = new_pandas_df

job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
job = client.load_table_from_dataframe(combined_df, table_id_bronze, job_config=job_config)
job.result()
print(f"Successfully uploaded {len(combined_df)} rows to {table_id_bronze}.")

# ==========================================
# 3. Cleaning Stage & Silver Layer
# ==========================================
# الفلترة الآمنة بدون مقارنة النص الفارغ مع الحقول الرقمية
df_bronze = df_bronze.filter(
    (col('id').isNotNull()) & 
    (col('id') > 0) & 
    (col('sku').isNotNull()) & 
    (col('sku') != '')
)

string_column_names = [field.name for field in df_bronze.schema.fields if isinstance(field.dataType, StringType)]
for column_name in string_column_names:
    df_bronze = df_bronze.withColumn(column_name, F.regexp_replace(F.col(column_name), r"\s+", " "))

df_silver = df_bronze.withColumn('brand',
                                 F.when(F.col('brand').isNull(), 'Unkown')
                                  .otherwise(F.col('brand')))

brand_mapping = {
    "Guchi": "Gucci", "Guzzi": "Gucci", "Chanell": "Chanel",
    "Esence": "Essence", "Diorr": "Dior", "Calvin": "Calvin Klein"
}
df_silver = df_silver.replace(brand_mapping, subset=["brand"])

info_and_desc_columns = ['description', 'warrantyInformation', 'shippingInformation']
for column_name in info_and_desc_columns:
    df_silver = df_silver.withColumn(column_name, F.when(F.col(column_name).isNull(), 'Unkown').otherwise(F.col(column_name)))

df_bronze = df_bronze.withColumn('price',
                                 F.when(F.col('price') < 0, F.abs(F.col('price')))
                                  .otherwise(F.col('price')))

df_bronze = df_bronze.withColumn('created_at', F.to_date(F.to_timestamp(F.col('created_at'), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")))
df_bronze = df_bronze.withColumn('updated_at', F.to_date(F.to_timestamp(F.col('updated_at'), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")))

df_silver = df_silver.withColumn("silver_loaded_at", current_timestamp())
table_id_silver = f"{project_id}.silver.products"

new_pandas_df_silver = df_silver.toPandas()

try:
    query = f"SELECT * FROM `{table_id_silver}`"
    old_pandas_df = client.query(query).to_dataframe()
    combined_silver = pd.concat([old_pandas_df, new_pandas_df_silver], ignore_index=True)
    combined_silver = combined_silver.drop_duplicates(subset=['product_id'], keep='last')
    print("Successfully merged old Silver data with new data locally.")
except Exception as e:
    print(f"Silver table does not exist yet or first run. Using new data only. Error: {e}")
    combined_silver = new_pandas_df_silver

job = client.load_table_from_dataframe(combined_silver, table_id_silver, job_config=job_config)
job.result()
print(f"Successfully uploaded {len(combined_silver)} rows to {table_id_silver}.")

# ==========================================
# 4. Gold Layer
# ==========================================
df_gold = df_silver.withColumn('final_price', F.round(F.col('price') - (F.col('price') * (F.col('discountPercentage') / 100)), 2))
df_gold = df_gold.withColumn('inventory_value', F.round(F.col('stock') * F.col('price'), 2))
df_gold = df_gold.withColumn('product_volume', F.round(F.col('dimension_width') * F.col('dimension_height') * F.col('dimension_depth'), 2))
df_gold = df_gold.withColumn('gold_loaded_at', current_timestamp())

dim_products = df_gold.select([
    'product_id', 'id', 'sku', 'price', 'discountPercentage', 'final_price',
    'stock', 'inventory_value', 'rating', 'weight', 'product_volume',
    'minimumOrderQuantity', 'gold_loaded_at', 'silver_loaded_at', 'bronze_loaded_at'
])

dim_products_profile = df_gold.select([
    'product_id', 'id', 'title', 'category', 'brand', 'all_tags',
    'created_at', 'description', 'gold_loaded_at', 'silver_loaded_at', 'bronze_loaded_at'
]).distinct()

dim_prods_logistics_policies = df_gold.select([
    'product_id', 'id', 'barcode', 'availabilityStatus', 'shippingInformation',
    'warrantyInformation', 'returnPolicy', 'dimension_width', 'dimension_depth',
    'dimension_height', 'gold_loaded_at', 'silver_loaded_at', 'bronze_loaded_at'
]).distinct()

gold_tables = {
    'dim_products': dim_products,
    'dim_products_profile': dim_products_profile,
    'dim_products_logistics_policies': dim_prods_logistics_policies
}

for table_name, df_table in gold_tables.items():
    table_id_gold = f"{project_id}.gold.{table_name}"
    new_pandas_df = df_table.toPandas()
    
    try:
        query = f"SELECT * FROM `{table_id_gold}`"
        old_pandas_df = client.query(query).to_dataframe()
        combined_gold = pd.concat([old_pandas_df, new_pandas_df], ignore_index=True)
        combined_gold = combined_gold.drop_duplicates(subset=['product_id'], keep='last')
        print(f"Successfully merged old data for {table_name}.")
    except Exception as e:
        print(f"Table {table_name} does not exist yet. Using new data only. Error: {e}")
        combined_gold = new_pandas_df

    job = client.load_table_from_dataframe(combined_gold, table_id_gold, job_config=job_config)
    job.result()
    print(f"Successfully uploaded {len(combined_gold)} rows to {table_id_gold}.")
