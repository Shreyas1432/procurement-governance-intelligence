WITH contract_price AS (
    SELECT
        contract_id,
        MAX(CASE WHEN consensus_flag THEN 1 ELSE 0 END) AS price_anomaly
    FROM rq3_anomalies
    GROUP BY contract_id
),
labeled AS (
    SELECT
        CASE WHEN p.risk_class = 'HIGH' THEN 'High governance risk' ELSE 'Other governance risk' END AS governance_group,
        CASE WHEN COALESCE(cp.price_anomaly, 0) = 1 THEN 'Price anomaly' ELSE 'No price anomaly' END AS price_group
    FROM rq2_predictions p
    LEFT JOIN contract_price cp USING (contract_id)
)
SELECT
    governance_group,
    price_group,
    COUNT(*) AS contracts
FROM labeled
GROUP BY governance_group, price_group
ORDER BY contracts DESC;
