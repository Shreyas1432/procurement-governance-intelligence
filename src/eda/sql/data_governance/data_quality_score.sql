WITH checks AS (
    SELECT
        metric,
        value,
        MAX(CASE WHEN metric = 'total_rows' THEN value END) OVER () AS total_rows
    FROM data_quality
),
rates AS (
    SELECT
        metric,
        CASE
            WHEN metric LIKE 'missing_%' THEN 1 - value / NULLIF(total_rows, 0)
            WHEN metric LIKE 'invalid_%' THEN 1 - value / NULLIF(total_rows, 0)
            ELSE NULL
        END AS quality_rate
    FROM checks
)
SELECT
    COALESCE(metric, 'overall') AS quality_dimension,
    quality_rate
FROM rates
WHERE quality_rate IS NOT NULL
UNION ALL
SELECT 'overall', AVG(quality_rate)
FROM rates
WHERE quality_rate IS NOT NULL
ORDER BY quality_dimension;
