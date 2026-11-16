# -*- coding: utf-8 -*-
import json
import requests
from datetime import datetime
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, current_timestamp, trim, to_date, coalesce
from pyspark.sql.types import StringType, NumericType
from google.cloud import bigquery

# ==========================================
# 1. Export Data From Api
# ==========================================
spark = SparkSession.builder \
    .appName("UsersAPItoSpark") \
    .getOrCreate()

# Disable ANSI mode for soft parsing on invalid dates
spark.conf.set("spark.sql.ansi.enabled", "false")

url = 'https://dummyjson.com/users'
response = requests.get(url)

if response.status_code == 200:
    json_data = response.json()
    users_list = json_data["users"]

    json_strings = [json.dumps(item) for item in users_list]
    rdd = spark.sparkContext.parallelize(json_strings)
    raw_df = spark.read.json(rdd)
else:
    print('Error Connection')

raw_df.printSchema()
raw_df.show(5)

# ==========================================
# 2. Bronze Layer
# ==========================================
# Unpacking nested structs and dicts
df_bronze = raw_df.select(
    F.col("id"),
    F.col("firstName"),
    F.col("lastName"),
    F.col("maidenName"),
    F.col("age"),
    F.col("gender"),
    F.col("email"),
    F.col("phone"),
    F.col("username"),
    F.col("birthDate"),
    F.col("image"),
    F.col("bloodGroup"),
    F.col("height"),
    F.col("weight"),
    F.col("eyeColor"),
    F.col("role"),
    F.col("ssn"),
    F.col("ein"),
    F.col("ip"),
    F.col("macAddress"),
    F.col("university"),
    F.col("userAgent"),

    # Address Unpacking
    F.col("address.address").alias("address_street"),
    F.col("address.city").alias("address_city"),
    F.col("address.state").alias("address_state"),
    F.col("address.stateCode").alias("address_state_code"),
    F.col("address.postalCode").alias("address_postal_code"),
    F.col("address.country").alias("address_country"),
    F.col("address.coordinates.lat").alias("address_lat"),
    F.col("address.coordinates.lng").alias("address_lng"),

    # Bank Unpacking
    F.col("bank.cardType").alias("bank_card_type"),
    F.col("bank.cardNumber").alias("bank_card_number"),
    F.col("bank.cardExpire").alias("bank_card_expire"),
    F.col("bank.currency").alias("bank_currency"),
    F.col("bank.iban").alias("bank_iban"),

    # Company Unpacking
    F.col("company.name").alias("company_name"),
    F.col("company.department").alias("company_department"),
    F.col("company.title").alias("company_title"),
    F.col("company.address.address").alias("company_address_street"),
    F.col("company.address.city").alias("company_address_city"),
    F.col("company.address.state").alias("company_address_state"),
    F.col("company.address.country").alias("company_address_country"),

    # Crypto Unpacking
    F.col("crypto.coin").alias("crypto_coin"),
    F.col("crypto.wallet").alias("crypto_wallet"),
    F.col("crypto.network").alias("crypto_network"),

    # Hair Unpacking
    F.col("hair.color").alias("hair_color"),
    F.col("hair.type").alias("hair_type")
)

# Add bronze_loaded_at timestamp
df_bronze = df_bronze.withColumn("bronze_loaded_at", current_timestamp())

# Create unique surrogate key for tracking and deduplication
df_bronze = df_bronze.withColumn(
    'user_surrogate_id',
    F.concat(F.col("id"), F.lit("_"), F.col("bronze_loaded_at").cast("string"))
)

project_id = "dummyjson-e2e-pipeline"
table_id_bronze = f"{project_id}.bronze.users"
client = bigquery.Client(project=project_id)

new_pandas_df = df_bronze.toPandas()

try:
    query = f"SELECT * FROM `{table_id_bronze}`"
    old_pandas_df = client.query(query).to_dataframe()
    combined_df = pd.concat([old_pandas_df, new_pandas_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['user_surrogate_id'], keep='last')
    print("Successfully merged old bronze data with new data locally.")
except Exception as e:
    print(f"Bronze table does not exist yet or first run. Using new data only. Error: {e}")
    combined_df = new_pandas_df

# Ensure explicit string datatype for Arrow/BigQuery compatibility
combined_df['user_surrogate_id'] = combined_df['user_surrogate_id'].astype(str)

job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
job = client.load_table_from_dataframe(combined_df, table_id_bronze, job_config=job_config)
job.result()
print(f"Successfully uploaded {len(combined_df)} rows to {table_id_bronze}.")

# ==========================================
# 3. Cleaning Stage & Silver Layer
# ==========================================
# Filter primary keys and core fields safely
df_bronze = df_bronze.filter(
    (col('id').isNotNull()) & 
    (col('id') > 0) & 
    (col('email').isNotNull()) & 
    (col('email') != '')
)

df_bronze = df_bronze.dropDuplicates(['id'])

# Trim string columns
string_column_names = [field.name for field in df_bronze.schema.fields if isinstance(field.dataType, StringType)]
for column_name in string_column_names:
    df_bronze = df_bronze.withColumn(column_name, F.regexp_replace(F.col(column_name), r"\s+", " "))

# IP Address Validation
ip_regex = r"^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
df_bronze = df_bronze.withColumn(
    "ip",
    F.when(F.col("ip").rlike(ip_regex), F.col("ip")).otherwise(F.lit(None))
)

# Replace missing textual fields with 'Unknown'
unknown_fill_columns = [
    'company_name', 'company_department', 'company_title', 'company_address_street',
    'company_address_city', 'company_address_state', 'company_address_country',
    'university', 'maidenName', 'hair_color', 'hair_type', 'crypto_coin',
    'crypto_wallet', 'crypto_network'
]
for col_name in unknown_fill_columns:
    df_bronze = df_bronze.withColumn(
        col_name,
        F.when((F.col(col_name).isNull()) | (F.col(col_name) == ''), "Unknown")
         .otherwise(F.col(col_name))
    )

# MAC Address Validation
mac_regex = r"^([0-9a-fA-F]{2}[:-]){5}([0-9a-fA-F]{2})$"
df_bronze = df_bronze.withColumn(
    "macAddress",
    F.when(F.col("macAddress").rlike(mac_regex), F.col("macAddress")).otherwise(F.lit(None))
)

# Blood Group Standardization
valid_blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
df_bronze = df_bronze.withColumn(
    'bloodGroup',
    F.when(F.col('bloodGroup').isin(valid_blood_groups), F.col('bloodGroup')).otherwise(F.lit('Unknown'))
)

# Phone Number Validation
phone_pattern = r'^\+\d{1,3}\s\d{3}-\d{3}-\d{4}$'
df_bronze = df_bronze.withColumn(
    'phone',
    F.when(F.col('phone').rlike(phone_pattern), F.col('phone')).otherwise(F.lit(None))
)

# Email Validation
email_regex = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
df_bronze = df_bronze.withColumn(
    "email",
    F.when(F.col("email").rlike(email_regex), F.col("email")).otherwise(F.lit(None))
)

# Gender Standardization
df_bronze = df_bronze.withColumn(
    "gender",
    F.when(F.lower(F.col("gender")).isin(["male", "m"]), F.lit("male"))
     .when(F.lower(F.col("gender")).isin(["female", "f"]), F.lit("female"))
     .otherwise(F.lit("Unknown"))
)

# Clean Numeric Bounds (age, height, weight)
bounds = {'age': (0, 120), 'height': (40, 250), 'weight': (2, 300)}
for col_name, (min_value, max_value) in bounds.items():
    df_bronze = df_bronze.withColumn(
        col_name,
        F.when((F.col(col_name) < min_value) | (F.col(col_name) > max_value), F.lit(None))
         .otherwise(F.col(col_name))
    )

# Date Normalization
date_formats = [
    "yyyy-M-d", "yyyy-MM-dd", "yyyy/MM/dd", "MM/dd/yyyy",
    "dd/MM/yyyy", "dd-MM-yyyy", "MM-dd-yyyy", "MMM dd, yyyy",
    "dd-MMM-yyyy", "yyyyMMdd", "yyyy-MM-dd HH:mm:ss"
]
trimmed_date = trim(col("birthDate"))
date_expressions = [to_date(trimmed_date, fmt) for fmt in date_formats]
df_bronze = df_bronze.withColumn("birthDate", coalesce(*date_expressions))

# Silver Loaded At timestamp
df_silver = df_bronze.withColumn('silver_loaded_at', F.current_timestamp())
table_id_silver = f"{project_id}.silver.users"

new_pandas_df_silver = df_silver.toPandas()

try:
    query = f"SELECT * FROM `{table_id_silver}`"
    old_pandas_df = client.query(query).to_dataframe()
    combined_silver = pd.concat([old_pandas_df, new_pandas_df_silver], ignore_index=True)
    combined_silver = combined_silver.drop_duplicates(subset=['user_surrogate_id'], keep='last')
    print("Successfully merged old Silver data with new data locally.")
except Exception as e:
    print(f"Silver table does not exist yet or first run. Using new data only. Error: {e}")
    combined_silver = new_pandas_df_silver

combined_silver['user_surrogate_id'] = combined_silver['user_surrogate_id'].astype(str)

job = client.load_table_from_dataframe(combined_silver, table_id_silver, job_config=job_config)
job.result()
print(f"Successfully uploaded {len(combined_silver)} rows to {table_id_silver}.")

# ==========================================
# 4. Gold Layer (Star Schema Model)
# ==========================================
df_gold_base = df_silver.withColumn('gold_loaded_at', F.current_timestamp())

# Dimension User Profiles
dim_users = df_gold_base.select([
    'user_surrogate_id', 'id', 'firstName', 'lastName', 'maidenName',
    'age', 'gender', 'email', 'phone', 'username', 'birthDate',
    'bloodGroup', 'height', 'weight', 'eyeColor', 'role', 'image',
    'gold_loaded_at', 'silver_loaded_at', 'bronze_loaded_at'
])

# Dimension User Location
dim_users_location = df_gold_base.select([
    'user_surrogate_id', 'id', 'address_street', 'address_city',
    'address_state', 'address_state_code', 'address_postal_code',
    'address_country', 'address_lat', 'address_lng',
    'gold_loaded_at', 'silver_loaded_at', 'bronze_loaded_at'
])

# Dimension User Company
dim_users_company = df_gold_base.select([
    'user_surrogate_id', 'id', 'company_name', 'company_department',
    'company_title', 'company_address_street', 'company_address_country',
    'company_address_state', 'company_address_city',
    'gold_loaded_at', 'silver_loaded_at', 'bronze_loaded_at'
])

# Dimension User Banking
dim_users_banking = df_gold_base.select([
    'user_surrogate_id', 'id', 'bank_card_type', 'bank_card_number',
    'bank_card_expire', 'bank_currency', 'bank_iban',
    'gold_loaded_at', 'silver_loaded_at', 'bronze_loaded_at'
])

# Dimension User Digital Assets
dim_users_digital_assets = df_gold_base.select([
    'user_surrogate_id', 'id', 'ssn', 'ein', 'ip', 'macAddress',
    'university', 'userAgent', 'crypto_coin', 'crypto_wallet', 'crypto_network',
    'gold_loaded_at', 'silver_loaded_at', 'bronze_loaded_at'
])

gold_tables = {
    'dim_users': dim_users,
    'dim_users_location': dim_users_location,
    'dim_users_company': dim_users_company,
    'dim_users_banking': dim_users_banking,
    'dim_users_digital_assets': dim_users_digital_assets
}

for table_name, df_table in gold_tables.items():
    table_id_gold = f"{project_id}.gold.{table_name}"
    new_pandas_df = df_table.toPandas()
    
    try:
        query = f"SELECT * FROM `{table_id_gold}`"
        old_pandas_df = client.query(query).to_dataframe()
        combined_gold = pd.concat([old_pandas_df, new_pandas_df], ignore_index=True)
        combined_gold = combined_gold.drop_duplicates(subset=['user_surrogate_id'], keep='last')
        print(f"Successfully merged old data for {table_name}.")
    except Exception as e:
        print(f"Table {table_name} does not exist yet. Using new data only. Error: {e}")
        combined_gold = new_pandas_df

    combined_gold['user_surrogate_id'] = combined_gold['user_surrogate_id'].astype(str)

    job = client.load_table_from_dataframe(combined_gold, table_id_gold, job_config=job_config)
    job.result()
    print(f"Successfully uploaded {len(combined_gold)} rows to {table_id_gold}.")
