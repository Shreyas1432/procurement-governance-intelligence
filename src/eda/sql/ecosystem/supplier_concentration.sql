SELECT
    supplier_name,
    contract_count,
    total_value,
    total_value / SUM(total_value) OVER () AS value_share
FROM supplier_master
WHERE total_value > 0
ORDER BY total_value DESC
LIMIT 25;
