SELECT
    award_year,
    COUNT(*) AS contract_count,
    COUNT(DISTINCT buyer_id) AS buyers,
    COUNT(DISTINCT supplier_id) AS suppliers,
    SUM(contract_value) AS total_value,
    AVG(CASE WHEN bid_count <= 1 THEN 1.0 ELSE 0.0 END) AS single_bid_rate
FROM contracts_base
WHERE award_year BETWEEN 2006 AND 2026
GROUP BY award_year
ORDER BY award_year;
