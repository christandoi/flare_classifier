import polars as pl
import psycopg

import read_spt3g_fits as spt3g

DB_URL = "postgresql://flare:flare@localhost:5432/flares"

# window grid settings, the same values used for the Python window fits
WINDOW_D = 12.0
STRIDE_D = 6.0
MIN_OBS = 6
FLARE_MARGIN_D = 6.0

# known flare not in the Tandoi et al. 2024 catalog: Guns et al. 2026 flare 5
EXTRA_FLARES = pl.DataFrame(
    {
        "gaia_id": ["6456528236185330816"],
        "mjd": [59083.43],
        "source": ["guns2026"],
    }
)

# output columns of dataset.window_features(), in table order
FEATURE_COLUMNS = [
    "ts",
    "fwhm_d",
    "t0_rel_d",
    "snr_amp_95",
    "snr_amp_150",
    "fwhm_at_bound",
    "n_good",
    "span_d",
    "null_chi2_red",
    "n_3sigma_both",
    "peak_snr",
]

INJECTION_COLUMNS = [
    "inj_t0_rel_d",
    "inj_fwhm_d",
    "inj_snr_peak",
    "inj_alpha",
    "inj_amp_95",
    "inj_amp_150",
    "inj_max_obs_snr_150",
    "inj_max_obs_snr_combined",
]


def connect(url=DB_URL):
    """
    Opens a connection to the flare database started by compose.yaml.

    Arguments
    ---------
    url : str
        PostgreSQL connection URL. Defaults to the local container.

    Returns
    -------
    conn : psycopg.Connection
        Open connection. Use it in a `with` block so it commits on success
        and rolls back on an error.
    """
    return psycopg.connect(url)


def copy_rows(conn, table, df):
    """
    Bulk-inserts a polars DataFrame into an existing table with COPY, which
    is much faster than one INSERT per row. Column names in `df` must match
    the table's column names; columns the table fills itself (serial ids)
    are left out of `df`.

    Arguments
    ---------
    conn : psycopg.Connection
        Open connection.
    table : str
        Name of the table to insert into.
    df : polars.DataFrame
        Rows to insert.

    Returns
    -------
    n_rows : int
        Number of rows inserted.
    """
    columns = ", ".join(df.columns)
    with conn.cursor() as cur:
        with cur.copy(f"COPY {table} ({columns}) FROM STDIN") as copy:
            for row in df.iter_rows():
                copy.write_row(row)
    return df.height


def run_sql_file(conn, path):
    """
    Runs every statement in a .sql file and returns the rows of the last one
    (e.g. the summary query at the end of build_windows.sql).

    Arguments
    ---------
    conn : psycopg.Connection
        Open connection.
    path : str
        Path to the .sql file.

    Returns
    -------
    rows : list of tuple
        Result rows of the last statement, or [] if it returns none.
    """
    with open(path) as f:
        sql = f.read()
    rows = []
    with conn.cursor() as cur:
        cur.execute(sql)  # several statements are allowed when there are no parameters
        # step through each statement's result, keeping the last one with rows
        while True:
            if cur.description:
                rows = cur.fetchall()
            if not cur.nextset():
                break
    return rows


def query_df(conn, sql):
    """
    Runs a query and returns the result as a polars DataFrame, with the
    column names from the query.

    Arguments
    ---------
    conn : psycopg.Connection
        Open connection.
    sql : str
        Query to run, e.g. 'SELECT * FROM dataset'.

    Returns
    -------
    df : polars.DataFrame
        One row per result row.
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        columns = [col.name for col in cur.description]
        rows = cur.fetchall()
    # infer_schema_length=None scans every row, so columns that start with
    # NULLs (e.g. inj_id for quiescent rows) still get the right type
    return pl.DataFrame(rows, schema=columns, orient="row", infer_schema_length=None)


def load_inputs(conn, lightcurves, catalog, extra_flares=EXTRA_FLARES):
    """
    Loads the raw inputs into the database: stars, observations, known
    flares and the window grid settings. Empties those tables first (and
    everything that depends on them), so it can be re-run safely.

    The window anchor is computed the same way as in the notebook: the
    first observation time over all stars, plus 2.5 days.

    Arguments
    ---------
    conn : psycopg.Connection
        Open connection.
    lightcurves : polars.DataFrame
        Output of read_lightcurves().
    catalog : polars.DataFrame
        Tandoi et al. 2024 flare catalog, with dr3_source_id read as a string.
    extra_flares : polars.DataFrame
        Known flares outside the catalog, with columns gaia_id, mjd, source.
        Defaults to Guns et al. 2026 flare 5.

    Returns
    -------
    counts : dict
        Number of rows inserted into each table.
    """
    stars = (
        lightcurves.select("gaia_id", "spt_id", "ra", "dec")
        .unique("gaia_id")
        .sort("gaia_id")
    )
    observations = lightcurves.select(
        "gaia_id", "mjd", "flux_95", "err_95", "flux_150", "err_150"
    )
    catalog_flares = pl.concat(
        [
            catalog.select(
                pl.col("dr3_source_id").alias("gaia_id"),
                "spt_id",
                "mjd",
                pl.col("ts").alias("ts_published"),
                "spectral_index_95_150",
                pl.lit("tandoi2024").alias("source"),
            ),
            extra_flares,
        ],
        how="diagonal",
    )  # diagonal: columns missing from extra_flares become null
    window_grid = pl.DataFrame(
        {
            "anchor": [lightcurves["mjd"].min() + 2.5],
            "window_d": [WINDOW_D],
            "stride_d": [STRIDE_D],
            "min_obs": [MIN_OBS],
            "flare_margin_d": [FLARE_MARGIN_D],
        }
    )

    with conn.cursor() as cur:
        cur.execute("TRUNCATE stars, observations, catalog_flares, window_grid CASCADE")

    counts = {}
    for table, df in [
        ("stars", stars),
        ("observations", observations),
        ("catalog_flares", catalog_flares),
        ("window_grid", window_grid),
    ]:
        counts[table] = copy_rows(conn, table, df)
    return counts


def attach_window_ids(conn, fits):
    """
    Adds the database window_id (and the window's contaminated flag) to fit
    results keyed by (gaia_id, window_start), as saved by the notebook.

    window_start is converted to the integer grid position window_index =
    round((window_start - anchor) / stride_d), so the join never compares
    floating-point start times. Raises an error if any fit has no matching
    window or if its window_start disagrees with the database's.

    Arguments
    ---------
    conn : psycopg.Connection
        Open connection; the windows table must already be built.
    fits : polars.DataFrame
        Fit results with gaia_id and window_start columns.

    Returns
    -------
    fits : polars.DataFrame
        The input rows plus window_id and contaminated.
    """
    with conn.cursor() as cur:
        anchor, stride_d = cur.execute(
            "SELECT anchor, stride_d FROM window_grid"
        ).fetchone()
        rows = cur.execute(
            "SELECT window_id, gaia_id, window_index, window_start, contaminated FROM windows"
        ).fetchall()
    windows = pl.DataFrame(
        rows,
        schema={
            "window_id": pl.Int32,
            "gaia_id": pl.String,
            "window_index": pl.Int32,
            "window_start_db": pl.Float64,
            "contaminated": pl.Boolean,
        },
        orient="row",
    )

    fits = fits.with_columns(
        ((pl.col("window_start") - anchor) / stride_d)
        .round(0)
        .cast(pl.Int32)
        .alias("window_index")
    ).join(windows, on=["gaia_id", "window_index"], how="left")

    N_unmatched = fits["window_id"].null_count()
    if N_unmatched > 0:
        raise ValueError(f"{N_unmatched} fits have no matching window in the database")
    max_start_diff = (fits["window_start"] - fits["window_start_db"]).abs().max()
    if max_start_diff > 1e-9:
        raise ValueError(
            f"window_start differs from the database by up to {max_start_diff} d"
        )

    return fits.drop("window_index", "window_start_db")


def load_window_fits(conn, null_fits):
    """
    Loads the TS fits of quiescent windows (fit_windows() output) into
    window_fits. Fits on windows that the database flags as contaminated
    are dropped, as the notebook did by joining null_fits to the quiet
    windows. Empties window_fits first.

    Arguments
    ---------
    conn : psycopg.Connection
        Open connection; the windows table must already be built.
    null_fits : polars.DataFrame
        Output of fit_windows(), keyed by gaia_id and window_start.

    Returns
    -------
    counts : dict
        N_input, N_dropped_contaminated and N_loaded.
    """
    fits = attach_window_ids(conn, null_fits)
    quiet = fits.filter(~pl.col("contaminated"))

    with conn.cursor() as cur:
        cur.execute("TRUNCATE window_fits")
    N_loaded = copy_rows(
        conn, "window_fits", quiet.select("window_id", *FEATURE_COLUMNS)
    )

    return {
        "N_input": fits.height,
        "N_dropped_contaminated": fits.height - quiet.height,
        "N_loaded": N_loaded,
    }


def load_injections(conn, inj_fits, injection_set="main", normalization="band_150"):
    """
    Loads injection fits (fit_injections() output) into the injections
    table under the name `injection_set`, replacing any earlier rows of that
    set. Every host window must be quiescent.

    Arguments
    ---------
    conn : psycopg.Connection
        Open connection; the windows table must already be built.
    inj_fits : polars.DataFrame
        Output of fit_injections(), keyed by inj_id, with gaia_id and
        window_start of the host window.
    injection_set : str
        Name of this run, e.g. 'main' or 'fixed_alpha'.
    normalization : str
        The normalization fit_injections() used: 'band_150' or 'combined'.

    Returns
    -------
    N_loaded : int
        Number of injections inserted.
    """
    if "inj_max_obs_snr_combined" not in inj_fits.columns:
        # older runs were made before this column existed
        inj_fits = inj_fits.with_columns(
            pl.lit(None, pl.Float64).alias("inj_max_obs_snr_combined")
        )

    fits = attach_window_ids(conn, inj_fits)
    N_contaminated_hosts = fits["contaminated"].sum()
    if N_contaminated_hosts > 0:
        raise ValueError(
            f"{N_contaminated_hosts} injections have a contaminated host window"
        )

    fits = fits.with_columns(
        pl.lit(injection_set).alias("injection_set"),
        pl.lit(normalization).alias("normalization"),
    )
    with conn.cursor() as cur:
        cur.execute("DELETE FROM injections WHERE injection_set = %s", (injection_set,))
    return copy_rows(
        conn,
        "injections",
        fits.select(
            "injection_set",
            "inj_id",
            "window_id",
            "normalization",
            *INJECTION_COLUMNS,
            *FEATURE_COLUMNS,
        ),
    )


if __name__ == "__main__":
    data_dir = "data"
    lightcurves_path = f"{data_dir}/spt3g_flare_lightcurves_2019_2022.fits"
    catalog_path = f"{data_dir}/spt3g_2019_2022_flare_star_catalog.csv"

    lightcurves = spt3g.read_lightcurves(lightcurves_path)
    catalog = pl.read_csv(catalog_path, schema_overrides={"dr3_source_id": pl.String})
    null_fits = pl.read_parquet(f"{data_dir}/null_fits.parquet")
    inj_fits = pl.read_parquet(f"{data_dir}/inj_fits.parquet")
    fixed_alpha_fits = pl.read_parquet(f"{data_dir}/fixed_alpha_test.parquet")

    # one transaction: if any step fails, nothing is changed
    with connect() as conn:
        print("inputs:", load_inputs(conn, lightcurves, catalog))
        print(
            "windows (N, N_quiescent, N_contaminated):",
            run_sql_file(conn, "sql/build_windows.sql"),
        )
        print("window_fits:", load_window_fits(conn, null_fits))
        print("injections (main):", load_injections(conn, inj_fits, "main", "band_150"))
        print(
            "injections (fixed_alpha):",
            load_injections(conn, fixed_alpha_fits, "fixed_alpha", "combined"),
        )
        print(
            "dataset (source, label, N):", run_sql_file(conn, "sql/build_dataset.sql")
        )
