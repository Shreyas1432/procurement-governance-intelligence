SELECT
    cpv_division,
    procedure_type,
    AVG(CASE WHEN bid_count <= 1 THEN 1 ELSE 0 END) AS single_bid_rate,
    COUNT(*) AS contracts
FROM rq2_features
WHERE cpv_division IS NOT NULL
GROUP BY cpv_division, procedure_type
HAVING COUNT(*) >= 100
ORDER BY single_bid_rate DESC
LIMIT 100;
