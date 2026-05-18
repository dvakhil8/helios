# Databricks notebook source
# MAGIC %md
# MAGIC # daily_revenue_by_country
# MAGIC
# MAGIC Aggregates order revenue by country and day. Reads from the seed catalog,
# MAGIC writes to the scratch run schema. Placeholders are substituted at job-
# MAGIC creation time by the eval harness (sandbox.py).
# MAGIC
# MAGIC NOTE: this query is deliberately suboptimal — see fixture.yaml.

# COMMAND ----------

SOURCE_ORDERS = "{{seed_catalog}}.{{seed_schema}}.orders"
SOURCE_USERS = "{{seed_catalog}}.{{seed_schema}}.users"
TARGET = "{{run_catalog}}.{{run_schema}}.{{output_table}}"

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE TABLE {TARGET} AS
    SELECT
        u.country,
        o.order_date,
        SUM(o.amount) AS revenue,
        COUNT(*) AS order_count
    FROM {SOURCE_ORDERS} o
    LEFT JOIN {SOURCE_USERS} u
        ON o.user_id = u.user_id
    GROUP BY u.country, o.order_date
""")
