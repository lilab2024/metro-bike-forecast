"""
Step 2 - Empirical metro-bike coupling diagnostics.

Two complementary tests are run over the buffer radii produced by
`01_build_station_panel.py`:

1. Bidirectional Granger causality between metro passenger flows and
   shared-bike usage, testing lags of up to five intervals in both
   directions. Reproduces Table 1 of the paper.

2. Station fixed-effect Negative Binomial GLMs relating metro entries to
   bike arrivals and metro exits to bike departures, fitted separately for
   each buffer. The coefficient profile across buffers identifies the
   catchment radius at which coupling is strongest. Reproduces Table 2.

Input layout (as written by step 1):

    <panel-root>/50m/merged_YYYYMMDD.csv
    <panel-root>/100m/merged_YYYYMMDD.csv
    ...

Usage
-----
    # Table 1 and Table 2 at 5-minute resolution
    python 02_coupling_diagnostics.py --panel-root outputs/panel --test all

    # Granger only, on the hourly panel
    python 02_coupling_diagnostics.py --panel-root outputs/panel \
        --test granger --resolution 1h
"""

from __future__ import annotations

import argparse
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

DEFAULT_BUFFERS = [50, 100, 200, 500]
DEFAULT_DATES = [f"202212{day:02d}" for day in range(12, 32)]

# Columns needed by both tests; anything else is dropped on load.
BASE_COLS = [
    "FDT_STARTTIME",
    "station__stationnames",
    "FINT_IN",
    "FINT_OUT",
    "ride_in",
    "ride_out",
]

# The hourly panel stores the same quantities under suffixed names.
HOURLY_RENAME = {
    "FINT_IN_1h": "FINT_IN",
    "FINT_OUT_1h": "FINT_OUT",
    "ride_in_1h": "ride_in",
    "ride_out_1h": "ride_out",
}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_buffer_panel(
    panel_root: Path, buffer_m: int, resolution: str, dates: list[str]
) -> pd.DataFrame:
    """Load every available day for one buffer radius into a single frame."""
    folder = panel_root / f"{buffer_m}m"

    if resolution == "1h":
        path = folder / "20days_data_1h.csv"
        if not path.exists():
            raise FileNotFoundError(f"Hourly panel not found: {path}")
        df = pd.read_csv(path).rename(columns=HOURLY_RENAME)
    else:
        files = [folder / f"merged_{d}.csv" for d in dates]
        files = [f for f in files if f.exists()]
        if not files:
            raise FileNotFoundError(f"No daily panels found in {folder}")
        frames = []
        for path in files:
            header = pd.read_csv(path, nrows=0).columns
            usecols = [c for c in BASE_COLS if c in header]
            frames.append(pd.read_csv(path, usecols=usecols))
            gc.collect()
        df = pd.concat(frames, ignore_index=True)

    for col in ["FINT_IN", "FINT_OUT", "ride_in", "ride_out"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")
    if "station__stationnames" in df.columns:
        df["station__stationnames"] = df["station__stationnames"].astype("category")
    return df


# --------------------------------------------------------------------------
# Test 1: bidirectional Granger causality
# --------------------------------------------------------------------------
def granger_both_directions(
    df: pd.DataFrame, metro_col: str, bike_col: str, maxlag: int = 5
) -> dict:
    """Minimum p-value and best lag for metro -> bike and bike -> metro.

    statsmodels tests whether the *second* column Granger-causes the first,
    so the column order is swapped between the two calls.
    """
    data = df[[metro_col, bike_col]].dropna()
    result = {}

    for label, columns in (
        ("metro->bike", [bike_col, metro_col]),
        ("bike->metro", [metro_col, bike_col]),
    ):
        try:
            tests = grangercausalitytests(data[columns], maxlag=maxlag, verbose=False)
            pvalues = [tests[lag + 1][0]["ssr_ftest"][1] for lag in range(maxlag)]
            result[f"{label}_p"] = min(pvalues)
            result[f"{label}_best_lag"] = int(np.argmin(pvalues)) + 1
        except Exception as exc:  # singular or degenerate series
            print(f"  Granger {label} failed for {metro_col}/{bike_col}: {exc}")
            result[f"{label}_p"] = np.nan
            result[f"{label}_best_lag"] = np.nan

    return result


def run_granger(
    panel_root: Path, buffers: list[int], resolution: str, dates: list[str], maxlag: int
) -> pd.DataFrame:
    rows = []
    for buffer_m in buffers:
        print(f"=== Granger causality, {buffer_m} m buffer ===")
        df = load_buffer_panel(panel_root, buffer_m, resolution, dates)

        entries = granger_both_directions(df, "FINT_IN", "ride_in", maxlag)
        exits = granger_both_directions(df, "FINT_OUT", "ride_out", maxlag)

        rows.append(
            {
                "Distance(m)": buffer_m,
                "FINT_IN->ride_in_p": entries["metro->bike_p"],
                "FINT_IN->ride_in_lag": entries["metro->bike_best_lag"],
                "ride_in->FINT_IN_p": entries["bike->metro_p"],
                "ride_in->FINT_IN_lag": entries["bike->metro_best_lag"],
                "FINT_OUT->ride_out_p": exits["metro->bike_p"],
                "FINT_OUT->ride_out_lag": exits["metro->bike_best_lag"],
                "ride_out->FINT_OUT_p": exits["bike->metro_p"],
                "ride_out->FINT_OUT_lag": exits["bike->metro_best_lag"],
            }
        )
        del df
        gc.collect()

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Test 2: station fixed-effect Negative Binomial GLM
# --------------------------------------------------------------------------
def fit_nb_fixed_effects(
    df: pd.DataFrame, response: str, predictor: str
) -> dict:
    """Fit `response ~ predictor + C(station)` with an NB2 error structure.

    Station dummies absorb time-invariant heterogeneity in land use, station
    function and built environment. BFGS is used because the dummy-expanded
    design matrix makes Newton steps expensive at the larger buffers.
    """
    subset = df.dropna(subset=[response, predictor, "station__stationnames"])
    subset = subset[subset[response] >= 0]

    formula = f"{response} ~ {predictor} + C(station__stationnames)"
    model = NegativeBinomial.from_formula(formula, data=subset).fit(
        method="bfgs", disp=0
    )
    return {
        "coef": float(model.params.get(predictor, np.nan)),
        "p": float(model.pvalues.get(predictor, np.nan)),
        "AIC": float(model.aic),
        "N": int(model.nobs),
    }


def run_nb_glm(
    panel_root: Path, buffers: list[int], resolution: str, dates: list[str]
) -> pd.DataFrame:
    rows = []
    for buffer_m in buffers:
        print(f"=== Negative Binomial GLM, {buffer_m} m buffer ===")
        df = load_buffer_panel(panel_root, buffer_m, resolution, dates)

        for pair, response, predictor in (
            ("IN", "FINT_IN", "ride_in"),
            ("OUT", "FINT_OUT", "ride_out"),
        ):
            try:
                stats = fit_nb_fixed_effects(df, response, predictor)
            except Exception as exc:
                print(f"  {pair} model failed at {buffer_m} m: {exc}")
                stats = {"coef": np.nan, "p": np.nan, "AIC": np.nan, "N": 0}
            rows.append({"Pair": pair, "Distance(m)": buffer_m, **stats})

        del df
        gc.collect()

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, default=Path("outputs/panel"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/diagnostics"))
    parser.add_argument(
        "--test", choices=["granger", "nbglm", "all"], default="all"
    )
    parser.add_argument(
        "--resolution",
        choices=["5min", "1h"],
        default="5min",
        help="5min reads merged_YYYYMMDD.csv; 1h reads 20days_data_1h.csv.",
    )
    parser.add_argument("--buffers", type=int, nargs="+", default=DEFAULT_BUFFERS)
    parser.add_argument("--maxlag", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.test in ("granger", "all"):
        granger = run_granger(
            args.panel_root, args.buffers, args.resolution, DEFAULT_DATES, args.maxlag
        )
        path = args.out_dir / f"granger_{args.resolution}.csv"
        granger.to_csv(path, index=False)
        print(f"\nGranger causality results -> {path}")
        print(granger.to_string(index=False))

    if args.test in ("nbglm", "all"):
        nb = run_nb_glm(args.panel_root, args.buffers, args.resolution, DEFAULT_DATES)
        path = args.out_dir / f"nb_glm_{args.resolution}.csv"
        nb.to_csv(path, index=False)
        print(f"\nNegative Binomial GLM results -> {path}")
        print(nb.to_string(index=False))


if __name__ == "__main__":
    main()
