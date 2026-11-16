# -*- coding: utf-8 -*-
import json
import requests
from datetime import datetime
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, current_timestamp, trim, to_timestamp
from pyspark.sql.types import StringType
from google.cloud import bigquery

# ==========================================
# 1. Export Data From Api (Open-Meteo Weather)
# ==========================================
spark = SparkSession.builder \
    .appName("WeatherAPItoSpark") \
    .getOrCreate()

url = 'https://api.open-meteo.com/v1/forecast'
params = {
    "latitude": 30.0444,
    "longitude": 31.2357,
    "current_weather": True
}

response = requests.get(url, params=params)

if response.status_code == 200:
    json_data = response.json()
    current_weather_data = json_data.get("current_weather", {})
    weather_list = [current_weather_data]

    json_strings = [json.dumps(item) for item in weather_list]
    rdd = spark.sparkContext.parallelize(json_strings)
    raw_df = spark.read.json(rdd)
else:
    print('Error Connection:', response.status_code)

raw_df.printSchema()
raw_df.show()

# ==========================================
# 2. Bronze Layer
# ==========================================
df_bronze = raw_df.select(
    F.col("time"),
    F.col("temperature"),
    F.col("windspeed"),
    F.col("winddirection"),
    F.col("weathercode"),
    F.col("is_day"),
    F.col("interval")
)

# Add bronze_loaded_at timestamp
df_bronze = df_bronze.withColumn("bronze_loaded_at", current_timestamp())

# Create unique surrogate key (weather_surrogate_id) for tracking and deduplication
df_bronze = df_bronze.withColumn(
    'weather_surrogate_id',
    F.concat(F.regexp_replace(F.col("time"), r"[^0-9]", ""), F.lit("_"), F.col("bronze_loaded_at").cast("string"))
)

project_id = "dummyjson-e2e-pipeline"
table_id_bronze = f"{project_id}.bronze.weather"
client = bigquery.Client(project=project_id)

new_pandas_df = df_bronze.toPandas()

try:
    query = f"SELECT * FROM `{table_id_bronze}`"
    old_pandas_df = client.query(query).to_dataframe()
    combined_df = pd.concat([old_pandas_df, new_pandas_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['weather_surrogate_id'], keep='last')
    print("Successfully merged old bronze data with new data locally.")
except Exception as e:
    print(f"Bronze table does not exist yet or first run. Using new data only. Error: {e}")
    combined_df = new_pandas_df

# Ensure explicit string datatype for Arrow/BigQuery compatibility
combined_df['weather_surrogate_id'] = combined_df['weather_surrogate_id'].astype(str)

job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
job = client.load_table_from_dataframe(combined_df, table_id_bronze, job_config=job_config)
job.result()
print(f"Successfully uploaded {len(combined_df)} rows to {table_id_bronze}.")

# ==========================================
# 3. Cleaning Stage & Silver Layer
# ==========================================
def clean_weather_data(df):
    # 1. Drop duplicates and nulls based on time
    df_clean = df.dropDuplicates(["time"])
    df_clean = df_clean.na.drop(subset=["time", "temperature"])

    # 2. Convert time string to proper Timestamp format
    df_clean = df_clean.withColumn(
        "weather_timestamp",
        F.to_timestamp(F.col("time"), "yyyy-MM-dd'T'HH:mm")
    )

    # 3. Clean numeric columns (Fix negative windspeed if any)
    df_clean = df_clean.withColumn(
        "windspeed",
        F.when(F.col("windspeed") < 0, 0.0).otherwise(F.col("windspeed"))
    )

    # 4. Add analytical category columns for Dashboard
    df_clean = df_clean.withColumn(
        "temp_category",
        F.when(F.col("temperature") >= 30, F.lit("Hot"))
         .when(F.col("temperature") >= 20, F.lit("Moderate"))
         .otherwise(F.lit("Cold"))
    )

    # Weather Code Description Mapping
    df_clean = df_clean.withColumn(
        "weather_condition",
        F.when(F.col("weathercode") == 0, F.lit("Clear Sky"))
         .when(F.col("weathercode").isin(1, 2, 3), F.lit("Cloudy"))
         .otherwise(F.lit("Other / Precipitation"))
    )

    # Wind Severity Mapping
    df_clean = df_clean.withColumn(
        "wind_severity",
        F.when(F.col("windspeed") < 10, F.lit("Calm"))
         .when(F.col("windspeed") <= 25, F.lit("Moderate Breeze"))
         .otherwise(F.lit("Strong Wind"))
    )

    # Day / Night Status Mapping
    df_clean = df_clean.withColumn(
        "day_night_status",
        F.when(F.col("is_day") == 1, F.lit("Daytime"))
         .otherwise(F.lit("Nighttime"))
    )

    # 5. Data Quality Audit Status
    df_clean = df_clean.withColumn(
        "calculation_audit_status",
        F.when(F.col("temperature").isNull() | F.col("windspeed").isNull(),
               F.lit("ERROR: Missing Critical Metrics"))
         .otherwise(F.lit("ALL_VALID"))
    )

    return df_clean

# Apply Cleaning
df_silver = clean_weather_data(df_bronze)

# Add silver_loaded_at timestamp
df_silver = df_silver.withColumn("silver_loaded_at", current_timestamp())
table_id_silver = f"{project_id}.silver.weather"

new_pandas_df_silver = df_silver.toPandas()

try:
    query = f"SELECT * FROM `{table_id_silver}`"
    old_pandas_df = client.query(query).to_dataframe()
    combined_silver = pd.concat([old_pandas_df, new_pandas_df_silver], ignore_index=True)
    combined_silver = combined_silver.drop_duplicates(subset=['weather_surrogate_id'], keep='last')
    print("Successfully merged old Silver data with new data locally.")
except Exception as e:
    print(f"Silver table does not exist yet or first run. Using new data only. Error: {e}")
    combined_silver = new_pandas_df_silver

combined_silver['weather_surrogate_id'] = combined_silver['weather_surrogate_id'].astype(str)

job = client.load_table_from_dataframe(combined_silver, table_id_silver, job_config=job_config)
job.result()
print(f"Successfully uploaded {len(combined_silver)} rows to {table_id_silver}.")

# ==========================================
# 4. Gold Layer (Star Schema Model)
# ==========================================
df_gold_base = df_silver.withColumn("gold_loaded_at", current_timestamp())

# Fact Weather Metrics Table
fact_weather_metrics = df_gold_base.select([
    'weather_surrogate_id',
    'weather_timestamp',
    'temperature',
    'windspeed',
    'winddirection',
    'temp_category',
    'wind_severity',
    'calculation_audit_status',
    'gold_loaded_at',
    'silver_loaded_at',
    'bronze_loaded_at'
])

# Dimension Weather Conditions Table
dim_weather_conditions = df_gold_base.select([
    'weather_surrogate_id',
    'weathercode',
    'weather_condition',
    'day_night_status',
    'interval',
    'gold_loaded_at',
    'silver_loaded_at',
    'bronze_loaded_at'
]).distinct()

# Dimension Location Information Table (Fixed Coordinates for Cairo)
dim_weather_location = df_gold_base.select([
    'weather_surrogate_id',
    F.lit(30.0444).alias("latitude"),
    F.lit(31.2357).alias("longitude"),
    F.lit("Cairo").alias("city"),
    F.lit("Egypt").alias("country"),
    'gold_loaded_at',
    'silver_loaded_at',
    'bronze_loaded_at'
]).distinct()

gold_tables = {
    'fact_weather_metrics': fact_weather_metrics,
    'dim_weather_conditions': dim_weather_conditions,
    'dim_weather_location': dim_weather_location
}

for table_name, df_table in gold_tables.items():
    table_id_gold = f"{project_id}.gold.{table_name}"
    new_pandas_df = df_table.toPandas()
    
    try:
        query = f"SELECT * FROM `{table_id_gold}`"
        old_pandas_df = client.query(query).to_dataframe()
        combined_gold = pd.concat([old_pandas_df, new_pandas_df], ignore_index=True)
        combined_gold = combined_gold.drop_duplicates(subset=['weather_surrogate_id'], keep='last')
        print(f"Successfully merged old data for {table_name}.")
    except Exception as e:
        print(f"Table {table_name} does not exist yet. Using new data only. Error: {e}")
        combined_gold = new_pandas_df

    combined_gold['weather_surrogate_id'] = combined_gold['weather_surrogate_id'].astype(str)

    job = client.load_table_from_dataframe(combined_gold, table_id_gold, job_config=job_config)
    job.result()
    print(f"Successfully uploaded {len(combined_gold)} rows to {table_id_gold}.")
