SELECT
    ROUND(log_price, 1) AS log_price_band,
    COUNT(*) AS contracts
FROM rq3_features
WHERE log_price IS NOT NULL
GROUP BY log_price_band
ORDER BY log_price_band;
