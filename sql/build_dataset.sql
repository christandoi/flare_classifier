-- Labeled training set, defined as views on the loaded tables.
-- SQL version of the notebook cells that built data/dataset.parquet.
--
-- Run with (after load_db.py has loaded the fits; load_db.py also runs it):
--   podman exec -i flare_db psql -U flare -d flares < sql/build_dataset.sql
--
-- Pipeline cuts (pipeline_cuts): pixel significance cut (at least one
-- observation above 3 sigma in both bands), TS > 45 and FWHM < 10 d.
--   ambiguous:              pipeline cuts + both amplitudes positive
--                           (quiescent windows that would count as detections;
--                           excluded from training, kept for the sensitivity check)
--   neg_pipeline_cuts_pass: pipeline cuts + both amplitudes negative
--                           (pure-noise reference for the false-positive rate)
--
-- dataset rows:
--   label 0, source 'quiescent': quiescent windows that are not ambiguous
--   label 1, source 'injection': main-run injections with observed peak
--                                SNR at 150 GHz >= 1 (observable)

DROP VIEW IF EXISTS dataset, ambiguous_windows, quiescent_fits;


-- every quiescent window fit, with its window and the pipeline-cut flags
CREATE VIEW quiescent_fits AS
SELECT
    w.window_id,
    w.gaia_id,
    w.window_start,
    wf.ts, wf.fwhm_d, wf.t0_rel_d, wf.snr_amp_95, wf.snr_amp_150, wf.fwhm_at_bound,
    wf.n_good, wf.span_d, wf.null_chi2_red, wf.n_3sigma_both, wf.peak_snr,
    cuts.pipeline_cuts,
    cuts.pipeline_cuts AND wf.snr_amp_95 > 0 AND wf.snr_amp_150 > 0 AS ambiguous,
    cuts.pipeline_cuts AND wf.snr_amp_95 < 0 AND wf.snr_amp_150 < 0 AS neg_pipeline_cuts_pass
FROM window_fits wf
JOIN windows w USING (window_id)
CROSS JOIN LATERAL (
    SELECT wf.n_3sigma_both >= 1 AND wf.ts > 45 AND wf.fwhm_d < 10 AS pipeline_cuts
) cuts;


-- the quiescent windows held out of training
CREATE VIEW ambiguous_windows AS
SELECT * FROM quiescent_fits WHERE ambiguous;


-- the classifier training set
CREATE VIEW dataset AS
SELECT
    q.window_id,
    q.gaia_id,
    q.window_start,
    NULL::integer AS inj_id,
    'quiescent' AS source,
    0 AS label,
    q.ts, q.fwhm_d, q.t0_rel_d, q.snr_amp_95, q.snr_amp_150, q.fwhm_at_bound,
    q.n_good, q.span_d, q.null_chi2_red, q.n_3sigma_both, q.peak_snr,
    q.neg_pipeline_cuts_pass,
    NULL::double precision AS inj_t0_rel_d,
    NULL::double precision AS inj_fwhm_d,
    NULL::double precision AS inj_snr_peak,
    NULL::double precision AS inj_alpha,
    NULL::double precision AS inj_amp_95,
    NULL::double precision AS inj_amp_150,
    NULL::double precision AS inj_max_obs_snr_150
FROM quiescent_fits q
WHERE NOT q.ambiguous

UNION ALL

SELECT
    w.window_id,
    w.gaia_id,
    w.window_start,
    i.inj_id,
    'injection' AS source,
    1 AS label,
    i.ts, i.fwhm_d, i.t0_rel_d, i.snr_amp_95, i.snr_amp_150, i.fwhm_at_bound,
    i.n_good, i.span_d, i.null_chi2_red, i.n_3sigma_both, i.peak_snr,
    NULL::boolean AS neg_pipeline_cuts_pass,
    i.inj_t0_rel_d, i.inj_fwhm_d, i.inj_snr_peak, i.inj_alpha,
    i.inj_amp_95, i.inj_amp_150, i.inj_max_obs_snr_150
FROM injections i
JOIN windows w USING (window_id)
WHERE i.injection_set = 'main'
    AND i.inj_max_obs_snr_150 >= 1;


-- summary: rows per class, plus the held-out ambiguous windows
SELECT source, label, count(*) AS n_rows
FROM dataset
GROUP BY source, label
UNION ALL
SELECT 'ambiguous (held out)', NULL, count(*) FROM ambiguous_windows
ORDER BY label NULLS LAST;