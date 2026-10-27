-- Schema for the flare classifier database.
--
-- Run with:
--   podman exec -i flare_db psql -U flare -d flares < sql/schema.sql
--
-- Re-running drops and recreates every table, so the database can always be
-- rebuilt from scratch (load_db.py, then build_dataset.sql).
--
-- Data flow:
--   stars, observations, catalog_flares, window_grid   <- load_db.py (raw inputs)
--   windows                                            <- build_dataset.sql
--   window_fits, injections                            <- load_db.py (TS fits from Python)
--   dataset (view)                                     <- build_dataset.sql
--
-- Windows are identified by an integer window_id. Their position on the grid
-- is stored as an integer window_index (window_start = anchor + stride_d *
-- window_index), so joins never compare floating-point start times.

DROP VIEW IF EXISTS dataset CASCADE;
DROP TABLE IF EXISTS injections, window_fits, windows, window_grid,
    catalog_flares, observations, stars CASCADE;


-- One row per star in the public light-curve release (66 stars).
CREATE TABLE stars (
    gaia_id     text PRIMARY KEY,           -- Gaia DR3 source id, as text
    spt_id      text NOT NULL,
    ra          double precision NOT NULL,  -- deg
    dec         double precision NOT NULL   -- deg
);


-- One row per observation (long format, as from read_lightcurves()).
-- Masked observations keep their row; errors may be NaN or Infinity, which
-- Postgres double precision stores as is.
CREATE TABLE observations (
    gaia_id     text NOT NULL REFERENCES stars (gaia_id),
    mjd         double precision NOT NULL,  -- days
    flux_95     double precision,           -- mJy
    err_95      double precision,
    flux_150    double precision,
    err_150     double precision,
    PRIMARY KEY (gaia_id, mjd)
);


-- Known flares used to flag contaminated windows: the Tandoi et al. 2024
-- catalog plus Guns et al. 2026 flare 5 (not in the catalog).
CREATE TABLE catalog_flares (
    flare_id                serial PRIMARY KEY,
    gaia_id                 text NOT NULL REFERENCES stars (gaia_id),
    spt_id                  text,               -- event name (e.g. J212043.7-543755); NULL outside the catalog
    mjd                     double precision NOT NULL,
    ts_published            double precision,   -- NULL for flares outside the catalog
    spectral_index_95_150   double precision,
    source                  text NOT NULL       -- 'tandoi2024' or 'guns2026'
);


-- Sliding-window grid settings, one row. Kept in the database so the SQL
-- that builds windows uses exactly the values the Python fits used.
CREATE TABLE window_grid (
    anchor          double precision NOT NULL,  -- fixed_anchor (MJD)
    window_d        double precision NOT NULL,  -- 12
    stride_d        double precision NOT NULL,  -- 6
    min_obs         integer NOT NULL,           -- 6
    flare_margin_d  double precision NOT NULL   -- 6
);


-- One row per kept window (built in SQL from observations + catalog_flares).
CREATE TABLE windows (
    window_id       serial PRIMARY KEY,
    gaia_id         text NOT NULL REFERENCES stars (gaia_id),
    window_index    integer NOT NULL,
    window_start    double precision NOT NULL,  -- MJD
    n_obs           integer NOT NULL,
    span_d          double precision NOT NULL,
    n_flares        integer NOT NULL,           -- known flares within the margin
    contaminated    boolean NOT NULL,           -- n_flares > 0
    UNIQUE (gaia_id, window_index)
);


-- TS fit and features of each quiescent window (output of fit_windows()).
CREATE TABLE window_fits (
    window_id       integer PRIMARY KEY REFERENCES windows (window_id),
    ts              double precision NOT NULL,
    fwhm_d          double precision NOT NULL,
    t0_rel_d        double precision NOT NULL,
    snr_amp_95      double precision NOT NULL,
    snr_amp_150     double precision NOT NULL,
    fwhm_at_bound   boolean NOT NULL,
    n_good          integer NOT NULL,
    span_d          double precision NOT NULL,  -- over unmasked observations
    null_chi2_red   double precision NOT NULL,
    n_3sigma_both   integer NOT NULL,
    peak_snr        double precision NOT NULL
);


-- Injected flares and their fits (output of fit_injections()).
-- injection_set separates the main run ('main') from the fixed-alpha test
-- ('fixed_alpha'). The fixed-alpha test reuses the same draws (inj_id) at
-- each alpha, so the key is (injection_set, inj_alpha, inj_id).
CREATE TABLE injections (
    injection_set               text NOT NULL,
    inj_id                      integer NOT NULL,
    window_id                   integer NOT NULL REFERENCES windows (window_id),
    normalization               text NOT NULL,      -- 'band_150' or 'combined'
    inj_t0_rel_d                double precision NOT NULL,
    inj_fwhm_d                  double precision NOT NULL,
    inj_snr_peak                double precision NOT NULL,
    inj_alpha                   double precision NOT NULL,
    inj_amp_95                  double precision NOT NULL,
    inj_amp_150                 double precision NOT NULL,
    inj_max_obs_snr_150         double precision NOT NULL,
    inj_max_obs_snr_combined    double precision,   -- NULL for runs made before this column existed
    ts                          double precision NOT NULL,
    fwhm_d                      double precision NOT NULL,
    t0_rel_d                    double precision NOT NULL,
    snr_amp_95                  double precision NOT NULL,
    snr_amp_150                 double precision NOT NULL,
    fwhm_at_bound               boolean NOT NULL,
    n_good                      integer NOT NULL,
    span_d                      double precision NOT NULL,
    null_chi2_red               double precision NOT NULL,
    n_3sigma_both               integer NOT NULL,
    peak_snr                    double precision NOT NULL,
    PRIMARY KEY (injection_set, inj_alpha, inj_id)
);

-- observations already has an index from its primary key (gaia_id, mjd)
CREATE INDEX ON catalog_flares (gaia_id, mjd);
CREATE INDEX ON injections (window_id);
