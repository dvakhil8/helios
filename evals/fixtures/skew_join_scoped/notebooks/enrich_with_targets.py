# Databricks notebook source
# MAGIC %md
# MAGIC # enrich_with_targets
# MAGIC OUT-OF-SCOPE. Adds a derived target column.

# COMMAND ----------

SOURCE = "{{run_catalog}}.{{run_schema}}.stg_daily_revenue"
TARGET = "{{run_catalog}}.{{run_schema}}.stg_enriched"

spark.sql(f"""
    CREATE OR REPLACE TABLE {TARGET} AS
    SELECT *, revenue * 1.10 AS target_revenue
    FROM {SOURCE}
""")
