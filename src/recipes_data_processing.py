# -*- coding: utf-8 -*-
import json
import requests
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, current_timestamp, trim
from pyspark.sql.types import StringType
from google.cloud import bigquery

# ==========================================
# 1. Export Data From Api
# ==========================================
spark = SparkSession.builder \
    .appName("RecipesAPItoSpark") \
    .getOrCreate()

url = 'https://dummyjson.com/recipes'
response = requests.get(url)

if response.status_code == 200:
    json_data = response.json()
    recipes_list = json_data["recipes"]

    json_strings = [json.dumps(item) for item in recipes_list]
    rdd = spark.sparkContext.parallelize(json_strings)
    raw_df = spark.read.json(rdd)
else:
    print('Error Connection')

raw_df.printSchema()
raw_df.show(5)

# ==========================================
# 2. Bronze Layer
# ==========================================
# Transform and Flatten the DataFrame
df_bronze = raw_df.select(
    F.col("id").alias("recipe_id"),
    F.col("name").alias("recipe_name"),
    F.col("cuisine"),
    F.col("difficulty"),
    F.col("caloriesPerServing"),
    F.col("prepTimeMinutes"),
    F.col("cookTimeMinutes"),
    F.col("servings"),
    F.col("rating"),
    F.col("reviewCount"),
    F.col("userId"),
    F.col("image").alias("image_url"),
    F.concat_ws(", ", F.col("ingredients")).alias("ingredients_list"),
    F.concat_ws(" | ", F.col("instructions")).alias("instructions_steps"),
    F.concat_ws(", ", F.col("mealType")).alias("meal_types"),
    F.concat_ws(", ", F.col("tags")).alias("recipe_tags")
)

# إضافة ختم البرونز وإنشاء المفتاح البديل اليومي الثابت (YYYY-MM-DD)
df_bronze = df_bronze.withColumn("bronze_loaded_at", current_timestamp())
df_bronze = df_bronze.withColumn(
    'recipe_surrogate_id',
    F.concat(
        F.col("recipe_id").cast("string"), 
        F.lit("_"), 
        F.date_format(F.col("bronze_loaded_at"), "yyyy-MM-dd")
    )
)

project_id = "dummyjson-e2e-pipeline"
table_id_bronze = f"{project_id}.bronze.recipes"
client = bigquery.Client(project=project_id)

new_pandas_df = df_bronze.toPandas()

try:
    query = f"SELECT * FROM `{table_id_bronze}`"
    old_pandas_df = client.query(query).to_dataframe()
    combined_df = pd.concat([old_pandas_df, new_pandas_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['recipe_surrogate_id'], keep='last')
    print("Successfully merged old bronze data with new data locally.")
except Exception as e:
    print(f"Bronze table does not exist yet or first run. Using new data only. Error: {e}")
    combined_df = new_pandas_df

combined_df['recipe_surrogate_id'] = combined_df['recipe_surrogate_id'].astype(str)

job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
job = client.load_table_from_dataframe(combined_df, table_id_bronze, job_config=job_config)
job.result()
print(f"Successfully uploaded {len(combined_df)} rows to {table_id_bronze}.")

# ==========================================
# 3. Cleaning Stage & Silver Layer
# ==========================================
# الفلترة الأولية للأعمدة على df_bronze
df_silver = df_bronze.filter(
    (col('recipe_id').isNotNull()) & 
    (col('recipe_id') > 0)
)

# تنظيف المسافات والنصوص
text_column_names = ['ingredients_list', 'instructions_steps', 'meal_types', 'difficulty', 'cuisine', 'recipe_tags', 'recipe_name']
for name in text_column_names:
    df_silver = df_silver.withColumn(
        name, trim(F.regexp_replace(F.col(name), r"\s+", " "))
    )

# تنظيف وتحسين الأعمدة الرقمية
df_silver = df_silver.withColumn(
    "caloriesPerServing",
    F.when((F.col("caloriesPerServing") == 0) | (F.col("caloriesPerServing").isNull()), F.lit(300))
     .otherwise(F.abs(F.col("caloriesPerServing")))
)

time_columns = ['cookTimeMinutes', 'prepTimeMinutes', 'servings']
for col_name in time_columns:
    df_silver = df_silver.withColumn(
        col_name,
        F.when((F.col(col_name) <= 0) | (F.col(col_name).isNull()), F.lit(1))
         .otherwise(F.col(col_name))
    )

# إضافة حالة التدقيق للحسابات
df_silver = df_silver.withColumn(
    'calculation_audit_status',
    F.when(~F.col('caloriesPerServing').between(10, 5000), F.lit('ERROR: Invalid Calories Range'))
     .when(~F.col('prepTimeMinutes').between(1, 1440), F.lit('ERROR: Invalid Prep Time'))
     .when(~F.col('cookTimeMinutes').between(0, 1440), F.lit('ERROR: Invalid Cook Time'))
     .otherwise(F.lit('ALL_VALID'))
)

# Feature Engineering للطبقة الفضية
df_silver = df_silver.withColumn(
    'rating_category',
    F.when(F.col('rating') >= 4.0, 'Good')
     .when(F.col('rating') >= 3.0, 'Average')
     .otherwise('Bad')
)

df_silver = df_silver.withColumn(
    'prep_time_category',
    F.when(F.col('prepTimeMinutes') >= 60, 'Long')
     .when(F.col('prepTimeMinutes') >= 30, 'Medium')
     .otherwise('Short')
)

df_silver = df_silver.withColumn(
    'calorie_level',
    F.when(F.col('caloriesPerServing') >= 500, 'High')
     .when(F.col('caloriesPerServing') >= 300, 'Medium')
     .otherwise('Low')
)

df_silver = df_silver.withColumn(
    'popularity_tier',
    F.when(F.col('reviewCount') >= 80, F.lit('High Popularity'))
     .when(F.col('reviewCount') >= 30, F.lit('Moderate'))
     .otherwise(F.lit('Low / New'))
)

df_silver = df_silver.withColumn(
    'serving_size_category',
    F.when(F.col('servings') <= 2, F.lit('Individual / Couple (1-2)'))
     .when(F.col('servings') <= 4, F.lit('Small Group (3-4)'))
     .otherwise(F.lit('Family Size (5+)'))
)

df_silver = df_silver.withColumn(
    'ingredient_count_category',
    F.when(F.size(F.split(F.col('ingredients_list'), ',')) < 5, F.lit('Few (<5)'))
     .when(F.size(F.split(F.col('ingredients_list'), ',')) <= 10, F.lit('Moderate (5-10)'))
     .otherwise(F.lit('Complex (10+)'))
)

# ختم زمن السيلفر
df_silver = df_silver.withColumn("silver_loaded_at", current_timestamp())
table_id_silver = f"{project_id}.silver.recipes"

new_pandas_df_silver = df_silver.toPandas()

try:
    query = f"SELECT * FROM `{table_id_silver}`"
    old_pandas_df = client.query(query).to_dataframe()
    combined_silver = pd.concat([old_pandas_df, new_pandas_df_silver], ignore_index=True)
    combined_silver = combined_silver.drop_duplicates(subset=['recipe_surrogate_id'], keep='last')
    print("Successfully merged old Silver data with new data locally.")
except Exception as e:
    print(f"Silver table does not exist yet or first run. Using new data only. Error: {e}")
    combined_silver = new_pandas_df_silver

combined_silver['recipe_surrogate_id'] = combined_silver['recipe_surrogate_id'].astype(str)

job = client.load_table_from_dataframe(combined_silver, table_id_silver, job_config=job_config)
job.result()
print(f"Successfully uploaded {len(combined_silver)} rows to {table_id_silver}.")

# ==========================================
# 4. Gold Layer (Star Schema Model)
# ==========================================
# نقل كامل البيانات المنظفة والمثرية إلى الجولد
df_gold = df_silver.withColumn("gold_loaded_at", current_timestamp())

# Fact Recipe Metrics
dim_recipe = df_gold.select([
    'recipe_surrogate_id',
    'recipe_id',
    'userId',
    'caloriesPerServing',
    'prepTimeMinutes',
    'cookTimeMinutes',
    'servings',
    'rating',
    'reviewCount',
    'rating_category',
    'prep_time_category',
    'calorie_level',
    'popularity_tier',
    'serving_size_category',
    'ingredient_count_category',
    'calculation_audit_status',
    'gold_loaded_at',
    'silver_loaded_at',
    'bronze_loaded_at'
])

# Dimension Recipes
dim_recipes = df_gold.select([
    'recipe_surrogate_id',
    'recipe_id',
    'recipe_name',
    'cuisine',
    'difficulty',
    'meal_types',
    'recipe_tags',
    'image_url',
    'gold_loaded_at',
    'silver_loaded_at',
    'bronze_loaded_at'
]).distinct()

# Dimension Recipe Instructions
dim_recipe_instructions = df_gold.select([
    'recipe_surrogate_id',
    'recipe_id',
    'ingredients_list',
    'instructions_steps',
    'gold_loaded_at',
    'silver_loaded_at',
    'bronze_loaded_at'
]).distinct()

gold_tables = {
    'dim_recipe': dim_recipe,
    'dim_recipes': dim_recipes,
    'dim_recipe_instructions': dim_recipe_instructions
}

for table_name, df_table in gold_tables.items():
    table_id_gold = f"{project_id}.gold.{table_name}"
    new_pandas_df = df_table.toPandas()
    
    try:
        query = f"SELECT * FROM `{table_id_gold}`"
        old_pandas_df = client.query(query).to_dataframe()
        combined_gold = pd.concat([old_pandas_df, new_pandas_df], ignore_index=True)
        combined_gold = combined_gold.drop_duplicates(subset=['recipe_surrogate_id'], keep='last')
        print(f"Successfully merged old data for {table_name}.")
    except Exception as e:
        print(f"Table {table_name} does not exist yet. Using new data only. Error: {e}")
        combined_gold = new_pandas_df

    combined_gold['recipe_surrogate_id'] = combined_gold['recipe_surrogate_id'].astype(str)

    job = client.load_table_from_dataframe(combined_gold, table_id_gold, job_config=job_config)
    job.result()
    print(f"Successfully uploaded {len(combined_gold)} rows to {table_id_gold}.")
