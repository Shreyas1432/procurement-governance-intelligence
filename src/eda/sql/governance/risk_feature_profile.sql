SELECT
    risk_class,
    COUNT(*) AS contracts,
    AVG(ensemble_risk_prob) AS mean_risk_probability,
    AVG(contract_amendments) AS mean_amendments
FROM rq2_predictions
GROUP BY risk_class
ORDER BY mean_risk_probability DESC;
