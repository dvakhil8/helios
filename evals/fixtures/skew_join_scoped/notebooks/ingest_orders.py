# Databricks notebook source
# MAGIC %md
# MAGIC # ingest_orders
# MAGIC OUT-OF-SCOPE. Simple CTAS from seed.

# COMMAND ----------

SOURCE = "{{seed_catalog}}.{{seed_schema}}.orders"
TARGET = "{{run_catalog}}.{{run_schema}}.stg_orders"

spark.sql(f"CREATE OR REPLACE TABLE {TARGET} AS SELECT * FROM {SOURCE}")
