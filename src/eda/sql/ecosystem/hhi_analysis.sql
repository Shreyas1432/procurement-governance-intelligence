WITH ranked AS (
    SELECT
        ROW_NUMBER() OVER (ORDER BY buyer_concentration_hhi) AS rank_id,
        COUNT(*) OVER () AS n,
        buyer_concentration_hhi
    FROM buyer_master
    WHERE buyer_concentration_hhi IS NOT NULL
)
SELECT
    rank_id,
    buyer_concentration_hhi,
    SUM(buyer_concentration_hhi) OVER (ORDER BY rank_id) / SUM(buyer_concentration_hhi) OVER () AS cumulative_hhi_share,
    CAST(rank_id AS DOUBLE) / NULLIF(n, 0) AS cumulative_buyer_share
FROM ranked
ORDER BY rank_id;
