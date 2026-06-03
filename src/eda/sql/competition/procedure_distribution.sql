SELECT
    procedure_type,
    COUNT(*) AS contracts,
    AVG(competition_risk) AS mean_competition_risk
FROM rq2_features
GROUP BY procedure_type
ORDER BY contracts DESC
LIMIT 20;
