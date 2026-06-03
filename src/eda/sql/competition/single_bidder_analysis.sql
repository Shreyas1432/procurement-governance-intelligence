SELECT
    procedure_type,
    COUNT(*) AS contracts,
    AVG(CASE WHEN bid_count <= 1 THEN 1 ELSE 0 END) AS single_bid_rate
FROM rq2_features
GROUP BY procedure_type
HAVING COUNT(*) >= 100
ORDER BY single_bid_rate DESC
LIMIT 20;
