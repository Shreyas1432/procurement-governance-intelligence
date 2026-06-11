WITH buyer_supplier_value AS (
    SELECT
        buyer_id,
        supplier_id,
        ANY_VALUE(buyer_name) AS buyer_name,
        SUM(contract_value) AS total_value,
        COUNT(*) AS contract_count,
        ROW_NUMBER() OVER (PARTITION BY buyer_id ORDER BY SUM(contract_value) DESC) AS supplier_rank
    FROM contracts_base
    WHERE buyer_id IS NOT NULL AND supplier_id IS NOT NULL AND contract_value > 0
    GROUP BY buyer_id, supplier_id
),
buyer_totals AS (
    SELECT buyer_id, SUM(total_value) AS buyer_total
    FROM buyer_supplier_value
    GROUP BY buyer_id
),
top5_value AS (
    SELECT buyer_id, SUM(total_value) AS top5_total
    FROM buyer_supplier_value
    WHERE supplier_rank <= 5
    GROUP BY buyer_id
)
SELECT
    bt.buyer_id,
    ANY_VALUE(bsv.buyer_name) AS buyer_name,
    bt.buyer_total,
    t5.top5_total,
    t5.top5_total / NULLIF(bt.buyer_total, 0) AS dependency_ratio,
    CASE
        WHEN t5.top5_total / NULLIF(bt.buyer_total, 0) > 0.8 THEN 'HIGH'
        WHEN t5.top5_total / NULLIF(bt.buyer_total, 0) > 0.5 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS dependency_class,
    COUNT(DISTINCT bsv.supplier_id) AS supplier_count,
    SUM(POWER(bsv.total_value / NULLIF(bt.buyer_total, 0), 2)) AS buyer_concentration_hhi
FROM buyer_totals bt
JOIN top5_value t5 ON bt.buyer_id = t5.buyer_id
JOIN buyer_supplier_value bsv ON bt.buyer_id = bsv.buyer_id
GROUP BY bt.buyer_id, bt.buyer_total, t5.top5_total
ORDER BY dependency_ratio DESC;
