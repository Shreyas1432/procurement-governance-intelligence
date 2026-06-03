SELECT
    CASE
        WHEN bid_count <= 1 THEN '0-1'
        WHEN bid_count <= 3 THEN '2-3'
        WHEN bid_count <= 5 THEN '4-5'
        WHEN bid_count <= 10 THEN '6-10'
        ELSE '11+'
    END AS bid_band,
    COUNT(*) AS contracts,
    AVG(competition_risk) AS mean_competition_risk
FROM rq2_features
GROUP BY bid_band
ORDER BY MIN(bid_count);
