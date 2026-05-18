# Databricks notebook source
# MAGIC %md
# MAGIC # ingest_users
# MAGIC OUT-OF-SCOPE. Simple CTAS from seed.

# COMMAND ----------

SOURCE = "{{seed_catalog}}.{{seed_schema}}.users"
TARGET = "{{run_catalog}}.{{run_schema}}.stg_users"

spark.sql(f"CREATE OR REPLACE TABLE {TARGET} AS SELECT * FROM {SOURCE}")
