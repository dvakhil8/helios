-- Same shape as skew_join_orders but smaller (3M rows) so the multi-task
-- run is fast enough to iterate on. The skew (87% NULL user_id) is
-- preserved — this fixture's purpose is scope architecture validation,
-- not dramatic perf demonstrations.

CREATE SCHEMA IF NOT EXISTS {{seed_catalog}}.{{seed_schema}};

DROP TABLE IF EXISTS {{seed_catalog}}.{{seed_schema}}.orders;
DROP TABLE IF EXISTS {{seed_catalog}}.{{seed_schema}}.users;

CREATE TABLE {{seed_catalog}}.{{seed_schema}}.users AS
SELECT
    id AS user_id,
    concat('user_', cast(id AS string)) AS user_name,
    element_at(array('US','UK','DE','FR','IN','JP','BR'), 1 + cast(id % 7 AS int)) AS country
FROM range(0, 50000);

CREATE TABLE {{seed_catalog}}.{{seed_schema}}.orders AS
SELECT
    id AS order_id,
    CASE
        WHEN rand(42) < 0.87 THEN NULL
        ELSE cast(rand(7) * 50000 AS bigint)
    END AS user_id,
    cast(rand(11) * 500 AS double) AS amount,
    date_add('2026-01-01', cast(id % 60 AS int)) AS order_date
FROM range(0, 3000000);
