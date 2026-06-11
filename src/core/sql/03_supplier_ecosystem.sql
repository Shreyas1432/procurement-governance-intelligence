SELECT
    supplier_id,
    ANY_VALUE(supplier_name) AS supplier_name,
    ANY_VALUE(supplier_region) AS supplier_region,
    COUNT(*) AS contract_count,
    COUNT(DISTINCT buyer_id) AS buyer_count,
    COUNT(DISTINCT cpv_division) AS cpv_diversity,
    MIN(award_year) AS first_year,
    MAX(award_year) AS last_year,
    SUM(contract_value) AS total_value,
    AVG(CASE WHEN bid_count <= 1 THEN 1.0 ELSE 0.0 END) AS single_bid_rate,
    AVG(CASE WHEN contract_amendments > 0 THEN 1.0 ELSE 0.0 END) AS amendment_rate
FROM contracts_base
WHERE supplier_id IS NOT NULL AND contract_value > 0
GROUP BY supplier_id
ORDER BY total_value DESC;
