"""
Step 1b - Aggregate the 5-minute station panel to hourly resolution.

Sums metro entries/exits and bike arrivals/departures within each
(line, station, date, hour) cell and writes one hourly panel per buffer
radius:

    <panel-root>/<d>m/20days_data_1h.csv

This file is the input of the 1-hour scenario in `03_forecast_global.py`
and of `04_forecast_station_level.py`.

Output columns: `station__FSTR_LINEID`, `station__FSTR_STATIONID`,
`station__stationnames`, `station__lon`, `station__lat`, `date`, `hour`,
`FINT_IN_1h`, `FINT_OUT_1h`, `ride_in_1h`, `ride_out_1h`.

Note on the aggregation
-----------------------
The daily panels written by step 1 carry one row per matched bike record,
and the window-level counts (`FINT_IN`, `FINT_OUT`, `ride_in`, `ride_out`)
are repeated identically across every row of the same station-window. The
sum below therefore adds each 5-minute count once per matched bike record
rather than once per window. This reproduces the aggregation used for the
published results; see the README for the implications.

The original aggregation concatenated all 20 days before grouping. Here the
daily files are grouped one at a time and the partial results concatenated,
which is arithmetically identical because `date` is a grouping key, and it
keeps peak memory proportional to a single day rather than the whole window.

Usage
-----
    python 01b_aggregate_hourly.py --panel-root outputs/panel \
        --buffers 50 100 200 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_BUFFERS = [50, 100, 200, 500]
DEFAULT_DATES = [f"202212{day:02d}" for day in range(12, 32)]

COUNT_COLS = ["FINT_IN", "FINT_OUT", "ride_in", "ride_out"]

GROUP_COLS = [
    "station__FSTR_LINEID",
    "station__FSTR_STATIONID",
    "station__stationnames",
    "station__lon",
    "station__lat",
    "date",
    "hour",
]

USECOLS = GROUP_COLS[:5] + ["FDT_STARTTIME"] + COUNT_COLS


def aggregate_day(path: Path) -> pd.DataFrame:
    """Aggregate one daily 5-minute panel to hourly cells."""
    # Read as strings first: station ids carry leading zeros that must survive.
    df = pd.read_csv(path, usecols=USECOLS, dtype=str)

    df["FDT_STARTTIME"] = pd.to_datetime(df["FDT_STARTTIME"], errors="coerce")
    for col in COUNT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["date"] = df["FDT_STARTTIME"].dt.date
    df["hour"] = df["FDT_STARTTIME"].dt.floor("h")

    return (
        df.groupby(GROUP_COLS)
        .agg(
            FINT_IN_1h=("FINT_IN", "sum"),
            FINT_OUT_1h=("FINT_OUT", "sum"),
            ride_in_1h=("ride_in", "sum"),
            ride_out_1h=("ride_out", "sum"),
        )
        .reset_index()
    )


def aggregate_buffer(panel_root: Path, buffer_m: int, dates: list[str]) -> Path | None:
    folder = panel_root / f"{buffer_m}m"
    files = [folder / f"merged_{d}.csv" for d in dates]
    files = [f for f in files if f.exists()]
    if not files:
        print(f"[skip] no daily panels found in {folder}")
        return None

    print(f"=== {buffer_m} m buffer: {len(files)} daily files ===")
    daily = [aggregate_day(path) for path in files]

    hourly = pd.concat(daily, ignore_index=True)
    hourly = hourly.sort_values(
        ["station__FSTR_LINEID", "station__FSTR_STATIONID", "hour"]
    ).reset_index(drop=True)

    out_path = folder / "20days_data_1h.csv"
    hourly.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(
        f"  {len(hourly)} hourly rows, "
        f"{hourly['station__stationnames'].nunique()} stations -> {out_path}"
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, default=Path("outputs/panel"))
    parser.add_argument("--buffers", type=int, nargs="+", default=DEFAULT_BUFFERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for buffer_m in args.buffers:
        aggregate_buffer(args.panel_root, buffer_m, DEFAULT_DATES)


if __name__ == "__main__":
    main()
