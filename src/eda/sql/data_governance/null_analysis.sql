WITH total AS (
    SELECT value AS total_rows
    FROM data_quality
    WHERE metric = 'total_rows'
),
missing AS (
    SELECT
        REPLACE(metric, 'missing_', '') AS field,
        value AS missing_rows
    FROM data_quality
    WHERE metric LIKE 'missing_%'
)
SELECT
    field,
    missing_rows,
    1 - missing_rows / NULLIF(total_rows, 0) AS completeness_rate
FROM missing
CROSS JOIN total
ORDER BY completeness_rate;
