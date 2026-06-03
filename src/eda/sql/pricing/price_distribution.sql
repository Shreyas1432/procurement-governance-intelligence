SELECT
    CASE
        WHEN final_price < 1000 THEN '<1k'
        WHEN final_price < 10000 THEN '1k-10k'
        WHEN final_price < 100000 THEN '10k-100k'
        WHEN final_price < 1000000 THEN '100k-1m'
        ELSE '1m+'
    END AS price_band,
    COUNT(*) AS contracts
FROM rq3_features
GROUP BY price_band
ORDER BY MIN(final_price);
