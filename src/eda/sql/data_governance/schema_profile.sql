SELECT
    column_type,
    COUNT(*) AS column_count
FROM schema_audit
GROUP BY column_type
ORDER BY column_count DESC;
