# Databricks notebook source
# MAGIC %md
# MAGIC # daily_revenue
# MAGIC IN-SCOPE. The bottleneck — deliberately suboptimal LEFT JOIN with
# MAGIC NULL-skewed user_id.

# COMMAND ----------

SOURCE_ORDERS = "{{run_catalog}}.{{run_schema}}.stg_orders"
SOURCE_USERS = "{{run_catalog}}.{{run_schema}}.stg_users"
TARGET = "{{run_catalog}}.{{run_schema}}.stg_daily_revenue"

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
