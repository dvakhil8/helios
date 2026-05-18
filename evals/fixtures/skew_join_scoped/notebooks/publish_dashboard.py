# Databricks notebook source
# MAGIC %md
# MAGIC # publish_dashboard
# MAGIC OUT-OF-SCOPE. Final projection for downstream consumption.

# COMMAND ----------

SOURCE = "{{run_catalog}}.{{run_schema}}.stg_enriched"
TARGET = "{{run_catalog}}.{{run_schema}}.dashboard_extract"

spark.sql(f"""
    CREATE OR REPLACE TABLE {TARGET} AS
    SELECT * FROM {SOURCE}
    WHERE country IS NOT NULL
    ORDER BY order_date
""")
