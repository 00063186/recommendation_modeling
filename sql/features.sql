-- Example 1: per-user behavioral aggregates from the training window.
-- :cutoff is the training/test boundary used by the feature pipeline.
SELECT
    user_id,
    COUNT(*) AS user_interaction_count,
    COUNT(CASE WHEN event_name = 'af_click' THEN 1 END) AS user_click_count,
    COUNT(CASE WHEN event_name = 'af_install' THEN 1 END) AS user_install_count,
    COUNT(CASE WHEN event_name = 'af_open' THEN 1 END) AS user_open_count,
    COUNT(CASE WHEN event_name = 'af_purchase' THEN 1 END) AS user_purchase_count,
    COUNT(CASE WHEN attributed_touch_type = 'impression' THEN 1 END) AS user_impression_count,
    AVG(duration_seconds) AS user_duration_avg
FROM interactions
WHERE event_time < :cutoff
GROUP BY user_id;


-- Example 2: per-app popularity / engagement aggregates from the same training window.
SELECT
    app_id,
    COUNT(*) AS app_interaction_count,
    COUNT(DISTINCT user_id) AS app_unique_users,
    COUNT(CASE WHEN event_name = 'af_click' THEN 1 END) AS app_click_count,
    COUNT(CASE WHEN event_name = 'af_install' THEN 1 END) AS app_install_count,
    COUNT(CASE WHEN event_name = 'af_open' THEN 1 END) AS app_open_count,
    COUNT(CASE WHEN event_name = 'af_purchase' THEN 1 END) AS app_purchase_count,
    AVG(duration_seconds) AS app_duration_avg
FROM interactions
WHERE event_time < :cutoff
GROUP BY app_id;
