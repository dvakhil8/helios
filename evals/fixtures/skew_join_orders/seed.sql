-- Seed data for skew_join_orders fixture.
--
-- Idempotent: drops and recreates the fixture's schema under the seed catalog.
-- Placeholders ({{seed_catalog}}) are substituted by sandbox.py at run time.
--
-- Tables produced:
--   {{seed_catalog}}.{{seed_schema}}.orders  - 10M rows, ~87% user_id NULL
--   {{seed_catalog}}.{{seed_schema}}.users   - 100K rows
--
-- The skew lives in orders.user_id. Aggregations that join through user_id
-- end up with one reducer handling nearly all NULL rows.

CREATE SCHEMA IF NOT EXISTS {{seed_catalog}}.{{seed_schema}};

DROP TABLE IF EXISTS {{seed_catalog}}.{{seed_schema}}.orders;
DROP TABLE IF EXISTS {{seed_catalog}}.{{seed_schema}}.users;

CREATE TABLE {{seed_catalog}}.{{seed_schema}}.users AS
SELECT
    id AS user_id,
    concat('user_', cast(id AS string)) AS user_name,
    element_at(array('US','UK','DE','FR','IN','JP','BR'), 1 + cast(id % 7 AS int)) AS country
FROM range(0, 100000);

CREATE TABLE {{seed_catalog}}.{{seed_schema}}.orders AS
SELECT
    id AS order_id,
    CASE
        WHEN rand(42) < 0.87 THEN NULL
        ELSE cast(rand(7) * 100000 AS bigint)
    END AS user_id,
    cast(rand(11) * 500 AS double) AS amount,
    date_add('2026-01-01', cast(id % 120 AS int)) AS order_date
FROM range(0, 10000000);
