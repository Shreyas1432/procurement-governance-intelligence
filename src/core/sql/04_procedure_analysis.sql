SELECT
    procedure_type,
    COUNT(*) AS total_contracts,
    SUM(CASE WHEN bid_count = 1 THEN 1 ELSE 0 END) AS single_bidder,
    100.0 * SUM(CASE WHEN bid_count = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS single_bidder_pct,
    AVG(bid_count) AS avg_bids,
    MEDIAN(final_price) AS median_price
FROM contracts_base
WHERE bid_count IS NOT NULL
GROUP BY procedure_type
ORDER BY single_bidder_pct DESC;
