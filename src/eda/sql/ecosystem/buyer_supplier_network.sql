SELECT
    buyer_id,
    supplier_id,
    COUNT(*) AS contract_count,
    SUM(contract_value) AS total_value
FROM rq1_features
GROUP BY buyer_id, supplier_id
ORDER BY total_value DESC
LIMIT 5000;
