-- Builds the windows table from observations, catalog_flares and window_grid.
-- SQL version of dataset.enumerate_windows() / star_windows().
--
-- Run with:
--   podman exec -i flare_db psql -U flare -d flares < sql/build_windows.sql
--
-- Rules (same as the Python version):
--   * grid: window_start = anchor + stride_d * window_index
--   * an observation is in a window if window_start <= mjd < window_start + window_d
--   * windows with fewer than min_obs observations are dropped
--   * a known flare counts toward a window if its mjd falls within
--     [window_start - flare_margin_d, window_start + window_d + flare_margin_d)
--
-- Re-running empties windows (and the fits/injections that point to it).

TRUNCATE windows RESTART IDENTITY CASCADE;

INSERT INTO windows (gaia_id, window_index, window_start, n_obs, span_d, n_flares, contaminated)
WITH
-- range of grid indices that could hold observations of each star; the -1
-- keeps the window that starts before the star's first observation
star_range AS (
    SELECT
        o.gaia_id,
        floor((min(o.mjd) - g.anchor) / g.stride_d)::integer - 1 AS k_min,
        floor((max(o.mjd) - g.anchor) / g.stride_d)::integer AS k_max
    FROM observations o
    CROSS JOIN window_grid g
    GROUP BY o.gaia_id, g.anchor, g.stride_d
),

-- every candidate window of every star
candidates AS (
    SELECT
        s.gaia_id,
        k AS window_index,
        g.anchor + g.stride_d * k AS window_start
    FROM star_range s
    CROSS JOIN window_grid g
    CROSS JOIN LATERAL generate_series(s.k_min, s.k_max) AS k
),

-- observation count and span per window, keeping windows with enough data
kept AS (
    SELECT
        c.gaia_id,
        c.window_index,
        c.window_start,
        count(*) AS n_obs,
        max(o.mjd) - min(o.mjd) AS span_d
    FROM candidates c
    CROSS JOIN window_grid g
    JOIN observations o
        ON o.gaia_id = c.gaia_id
        AND o.mjd >= c.window_start
        AND o.mjd < c.window_start + g.window_d
    GROUP BY c.gaia_id, c.window_index, c.window_start, g.min_obs
    HAVING count(*) >= g.min_obs
)

-- known flares within the margin of each kept window
SELECT
    k.gaia_id,
    k.window_index,
    k.window_start,
    k.n_obs,
    k.span_d,
    f.n_flares,
    f.n_flares > 0 AS contaminated
FROM kept k
CROSS JOIN window_grid g
CROSS JOIN LATERAL (
    SELECT count(*) AS n_flares
    FROM catalog_flares cf
    WHERE cf.gaia_id = k.gaia_id
        AND cf.mjd >= k.window_start - g.flare_margin_d
        AND cf.mjd < k.window_start + g.window_d + g.flare_margin_d
) f
ORDER BY k.gaia_id, k.window_index;

-- summary
SELECT
    count(*) AS n_windows,
    count(*) FILTER (WHERE NOT contaminated) AS n_quiescent,
    count(*) FILTER (WHERE contaminated) AS n_contaminated
FROM windows;