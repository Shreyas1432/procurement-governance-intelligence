WITH buyer_supplier_value AS (
    SELECT buyer_id, supplier_id, SUM(contract_value) AS supplier_value
    FROM contracts_base
    WHERE buyer_id IS NOT NULL AND supplier_id IS NOT NULL AND contract_value > 0
    GROUP BY buyer_id, supplier_id
),
shares AS (
    SELECT
        buyer_id,
        supplier_id,
        supplier_value,
        supplier_value / NULLIF(SUM(supplier_value) OVER (PARTITION BY buyer_id), 0) AS supplier_share
    FROM buyer_supplier_value
)
SELECT
    buyer_id,
    MAX(supplier_share) AS buyer_dependency_ratio,
    SUM(POWER(supplier_share, 2)) AS buyer_concentration_hhi,
    COUNT(*) AS buyer_supplier_count
FROM shares
GROUP BY buyer_id;
