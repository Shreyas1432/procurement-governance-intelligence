SELECT
    risk_class,
    COUNT(*) AS contracts,
    COUNT(*) / SUM(COUNT(*)) OVER () AS class_share
FROM rq2_predictions
GROUP BY risk_class
ORDER BY contracts DESC;
