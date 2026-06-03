SELECT
    dependency_class,
    COUNT(*) AS buyers,
    AVG(dependency_ratio) AS mean_dependency,
    AVG(buyer_concentration_hhi) AS mean_hhi
FROM buyer_master
GROUP BY dependency_class
ORDER BY mean_dependency DESC;
