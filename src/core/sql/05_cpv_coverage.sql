SELECT
    cpv_division,
    COUNT(*) AS record_count,
    COUNT(DISTINCT buyer_id) AS buyers,
    COUNT(DISTINCT supplier_id) AS suppliers,
    AVG(final_price) AS avg_price,
    MEDIAN(final_price) AS median_price,
    COUNT(*) >= 1000 AS high_coverage
FROM contracts_base
WHERE cpv_division IS NOT NULL AND cpv_division != 'UNKNOWN' AND final_price > 0
GROUP BY cpv_division
ORDER BY record_count DESC;
