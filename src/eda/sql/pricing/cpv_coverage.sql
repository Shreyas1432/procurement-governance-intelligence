SELECT
    cpv_division,
    COUNT(*) AS priced_contracts,
    AVG(final_price) AS mean_price,
    MEDIAN(final_price) AS median_price
FROM rq3_features
GROUP BY cpv_division
ORDER BY priced_contracts DESC
LIMIT 25;
