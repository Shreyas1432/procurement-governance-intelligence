WITH ranked AS (
    SELECT
        supplier_id,
        total_value,
        NTILE(4) OVER (ORDER BY total_value DESC) AS tier
    FROM supplier_master
    WHERE total_value > 0
)
SELECT
    'Tier ' || tier AS supplier_tier,
    COUNT(*) AS suppliers,
    SUM(total_value) AS total_value
FROM ranked
GROUP BY tier
ORDER BY tier;
