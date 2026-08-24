"""
Step 1 - Travel-chain reconstruction and metro/bike data integration.

Rebuilds continuous bicycle activity timelines from asynchronous lock/unlock
records, classifies riding and parking segments, matches bike events to metro
station exits within a set of Manhattan-distance buffers, and aggregates
ride-in / ride-out counts onto the 5-minute AFC time grid.

For every buffer radius d and every day in the analysis window the script
writes one file:

    <out_root>/<d>m/merged_YYYYMMDD.csv

These files are the input of `02_coupling_diagnostics.py` and of the
5-minute forecasting scenario in `03_forecast_global.py`.

Expected raw inputs
-------------------
bike_csv       Shared-bike lock/unlock log with columns
               BICYCLE_ID, DATA_TIME, LONGITUDE, LATITUDE, LOCK_STATUS
station_coords Metro station coordinates: stationnames, lon, lat
station_info   Station dictionary: FSTR_LINEID, FSTR_STATIONID, FSTR_STATIONNAME
flow_dir       Directory of daily AFC files named
               HIST_STO_5MFLOW_5M_YYYYMMDD.CSV with columns
               FSTR_LINEID, FSTR_STATIONID, FDT_STARTTIME, FDT_ENDTIME,
               FINT_IN, FINT_OUT

Usage
-----
    python 01_build_station_panel.py \
        --bike-csv       data/bike20221212-20230101.csv \
        --station-coords data/metro_station_coordinates.csv \
        --station-info   data/station_info.csv \
        --flow-dir       data/afc_5min \
        --out-root       outputs/panel \
        --buffers 50 100 200 500
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# WGS84 equatorial radius, used for the Web Mercator projection.
EARTH_RADIUS_M = 6378137.0
DEG = np.pi / 180.0

# Shanghai bounding box used to drop obviously corrupted GPS fixes.
LON_MIN, LON_MAX = 120.0, 122.0
LAT_MIN, LAT_MAX = 30.0, 32.0

# Segments shorter than this are treated as GPS jitter rather than trips.
MIN_SEGMENT_SECONDS = 10


def lonlat_to_mercator(lon_deg: np.ndarray, lat_deg: np.ndarray):
    """Project WGS84 degrees to Web Mercator metres (EPSG:3857)."""
    x = EARTH_RADIUS_M * lon_deg * DEG
    lat_rad = np.clip(lat_deg, -85.05112878, 85.05112878) * DEG
    y = EARTH_RADIUS_M * np.log(np.tan(np.pi / 4.0 + lat_rad / 2.0))
    return x, y


def load_static_tables(station_coords_path: Path, station_info_path: Path) -> pd.DataFrame:
    """Build the station reference table (ids, name, coordinates) shared by every day."""
    station_coords = pd.read_csv(station_coords_path, encoding="utf-8")
    station_coords = station_coords.drop_duplicates().reset_index(drop=True)

    station_info = pd.read_csv(station_info_path, encoding="gbk")
    station_info = station_info.rename(columns={"FSTR_STATIONNAME": "stationnames"})

    # Line and station ids are used as join keys against the AFC files, where
    # they may be stored with different dtypes or padding.
    for col in ("FSTR_LINEID", "FSTR_STATIONID"):
        station_info[col] = station_info[col].astype(str).str.strip()

    # One row per station with coordinates; the line name is not needed here.
    stations = (
        station_info.merge(station_coords, how="inner", on="stationnames")
        .drop(columns=["linename"], errors="ignore")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return stations


def reconstruct_travel_chain(bike_all: pd.DataFrame, day: datetime) -> pd.DataFrame:
    """Rebuild one day of continuous bicycle timelines.

    Lock-status events are asynchronous, so a mirrored copy of each record is
    generated one second earlier with the lock state inverted. Merging the two
    copies turns the point events into continuous state intervals. Midnight
    anchors are then added at both ends of each bicycle's day so that the first
    and last intervals are not truncated.
    """
    day_str = day.strftime("%Y/%m/%d")
    bikes = bike_all[bike_all["DATA_TIME"].astype(str).str.contains(day_str, na=False)].copy()

    bikes["DATA_TIME"] = pd.to_datetime(bikes["DATA_TIME"], errors="coerce")
    bikes = bikes.dropna(subset=["DATA_TIME"]).sort_values("DATA_TIME").reset_index(drop=True)

    # Replace the raw bicycle identifier with a compact integer id.
    bikes = bikes.sort_values("BICYCLE_ID").reset_index(drop=True)
    bike_ids = bikes["BICYCLE_ID"].drop_duplicates().to_frame().reset_index(drop=True)
    bike_ids["id"] = range(len(bike_ids))
    bikes = bikes.merge(bike_ids, on="BICYCLE_ID", how="left")

    # Some exports carry a mangled COMPANY_ID header; normalise it.
    for col in bikes.columns:
        if "COMPANY" in col.upper():
            bikes = bikes.rename(columns={col: "type"})
            break

    bikes = bikes[
        bikes["LONGITUDE"].between(LON_MIN, LON_MAX, inclusive="neither")
        & bikes["LATITUDE"].between(LAT_MIN, LAT_MAX, inclusive="neither")
    ].copy()

    # Mirrored records: one second earlier with the lock state reversed.
    mirrored = bikes.copy()
    mirrored["DATA_TIME"] = mirrored["DATA_TIME"] - timedelta(seconds=1)
    mirrored["LOCK_STATUS"] = 1 - mirrored["LOCK_STATUS"]

    events = pd.concat([mirrored, bikes], ignore_index=True)
    events["DATA_TIME"] = pd.to_datetime(events["DATA_TIME"], errors="coerce")
    events["DATE"] = events["DATA_TIME"].dt.date

    # Keep only bicycles with at least two valid records on that day.
    events = (
        events.groupby(["id", "DATE"]).filter(lambda g: len(g) > 1).reset_index(drop=True)
    )

    # Midnight anchors at both ends of the day.
    ordered = events.sort_values(["id", "DATA_TIME"])
    start = ordered.groupby("id").head(1).copy()
    start.loc[:, "DATA_TIME"] = pd.to_datetime(day.strftime("%Y-%m-%d 00:00:00"))
    end = ordered.groupby("id").tail(1).copy()
    end.loc[:, "DATA_TIME"] = pd.to_datetime(
        (day + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")
    )

    chain = pd.concat([events, start, end], ignore_index=True)
    chain["DATA_TIME"] = pd.to_datetime(chain["DATA_TIME"], errors="coerce")
    chain = (
        chain.dropna(subset=["DATA_TIME"])
        .sort_values(["id", "DATA_TIME"])
        .reset_index(drop=True)
    )

    # Attach the following record of the same bicycle to each row.
    grouped = chain.groupby("id")
    chain["id_next"] = grouped["id"].shift(-1)
    chain["DATA_TIME_next"] = grouped["DATA_TIME"].shift(-1)
    chain["LOCK_STATUS_next"] = grouped["LOCK_STATUS"].shift(-1)
    chain["LONGITUDE_next"] = grouped["LONGITUDE"].shift(-1)
    chain["LATITUDE_next"] = grouped["LATITUDE"].shift(-1)
    chain["dt_s"] = (chain["DATA_TIME_next"] - chain["DATA_TIME"]).dt.total_seconds()

    valid = chain["id"].eq(chain["id_next"]) & chain["dt_s"].gt(0)
    # Locked-to-locked spans are parking, unlocked-to-unlocked spans are riding.
    chain["stop_dt_s"] = chain["dt_s"].where(
        valid & chain["LOCK_STATUS"].eq(1) & chain["LOCK_STATUS_next"].eq(1)
    )
    chain["ride_dt_s"] = chain["dt_s"].where(
        valid & chain["LOCK_STATUS"].eq(0) & chain["LOCK_STATUS_next"].eq(0)
    )

    chain = chain[chain["id"] == chain["id_next"]].reset_index(drop=True)
    chain[["dt_s", "stop_dt_s", "ride_dt_s"]] = chain[
        ["dt_s", "stop_dt_s", "ride_dt_s"]
    ].fillna(0)
    return chain


def match_bikes_to_stations(
    chain: pd.DataFrame, stations: pd.DataFrame, buffer_m: float
) -> pd.DataFrame:
    """Pair bike events with every station within `buffer_m` Manhattan metres.

    Manhattan rather than Euclidean distance is used because it approximates
    walking distance in a dense gridded street network.
    """
    stations = stations.copy()
    bikes = chain.copy()

    stations["_xm"], stations["_ym"] = lonlat_to_mercator(
        stations["lon"].to_numpy(), stations["lat"].to_numpy()
    )
    bikes["_xm"], bikes["_ym"] = lonlat_to_mercator(
        bikes["LONGITUDE"].to_numpy(), bikes["LATITUDE"].to_numpy()
    )

    bx = bikes["_xm"].to_numpy()
    by = bikes["_ym"].to_numpy()

    cols = stations.columns.tolist()
    ix_xm, ix_ym = cols.index("_xm"), cols.index("_ym")

    matches = []
    for row in stations.to_numpy():
        xm, ym = row[ix_xm], row[ix_ym]
        mask = (np.abs(bx - xm) + np.abs(by - ym)) <= float(buffer_m)
        if not mask.any():
            continue
        hit = bikes.loc[mask].copy()
        for col, value in zip(cols, row):
            hit[f"station__{col}"] = value
        matches.append(hit)

    if not matches:
        return bikes.head(0).copy()

    pairs = pd.concat(matches, ignore_index=True)

    bx_m, by_m = lonlat_to_mercator(
        pairs["LONGITUDE"].to_numpy(), pairs["LATITUDE"].to_numpy()
    )
    pairs["distance MB"] = np.abs(bx_m - pairs["station___xm"].to_numpy()) + np.abs(
        by_m - pairs["station___ym"].to_numpy()
    )
    pairs = pairs.drop(columns=["_xm", "_ym", "station___xm", "station___ym"], errors="ignore")
    return pairs.sort_values(["id", "DATA_TIME"]).reset_index(drop=True)


def aggregate_to_afc_grid(pairs: pd.DataFrame, flow_5min: pd.DataFrame) -> pd.DataFrame:
    """Join the matched bike events onto the 5-minute AFC windows and count flows."""
    flow_5min = flow_5min.rename(
        columns={
            "FSTR_LINEID": "station__FSTR_LINEID",
            "FSTR_STATIONID": "station__FSTR_STATIONID",
        }
    )
    merged = flow_5min.merge(
        pairs, how="left", on=["station__FSTR_LINEID", "station__FSTR_STATIONID"]
    )
    merged = merged.sort_values(["station__stationnames", "FDT_STARTTIME"]).reset_index(
        drop=True
    )

    # Initial stock: bicycles already parked in the station area before the
    # first departure of the day. The frame is sorted by station and window, so
    # a grouped cumulative sum marks the rows preceding that first departure.
    station = merged["station__stationnames"]
    first_departure = merged["LOCK_STATUS"].eq(1) & merged["LOCK_STATUS_next"].eq(0)
    before_first = first_departure.groupby(station).cumsum().eq(0)
    parked = merged["LOCK_STATUS"].eq(1) & merged["LOCK_STATUS_next"].eq(1)
    merged["initial number"] = (
        (parked & before_first).astype(int).groupby(station).transform("sum")
    )

    for col in ["DATA_TIME", "DATA_TIME_next", "FDT_STARTTIME", "FDT_ENDTIME"]:
        merged[col] = pd.to_datetime(merged[col], errors="coerce")

    in_window = (merged["DATA_TIME"] >= merged["FDT_STARTTIME"]) & (
        merged["DATA_TIME"] <= merged["FDT_ENDTIME"]
    )
    merged = merged[in_window].reset_index(drop=True)

    merged["dt_s"] = (merged["DATA_TIME_next"] - merged["DATA_TIME"]).dt.total_seconds()
    in_window = (merged["DATA_TIME"] >= merged["FDT_STARTTIME"]) & (
        merged["DATA_TIME"] <= merged["FDT_ENDTIME"]
    )
    long_enough = merged["dt_s"] > MIN_SEGMENT_SECONDS

    # ride_in  = unlock -> lock  (a bicycle arrives at the station area)
    # ride_out = lock  -> unlock (a bicycle departs from the station area)
    merged["_ride_in_flag"] = (
        merged["LOCK_STATUS"].eq(0)
        & merged["LOCK_STATUS_next"].eq(1)
        & long_enough
        & in_window
    ).astype(int)
    merged["_ride_out_flag"] = (
        merged["LOCK_STATUS"].eq(1)
        & merged["LOCK_STATUS_next"].eq(0)
        & long_enough
        & in_window
    ).astype(int)

    keys = ["station__stationnames", "FDT_STARTTIME", "FDT_ENDTIME"]
    merged["ride_in"] = merged.groupby(keys)["_ride_in_flag"].transform("sum")
    merged["ride_out"] = merged.groupby(keys)["_ride_out_flag"].transform("sum")
    merged["delta_bikes"] = merged["ride_in"] - merged["ride_out"]
    return merged.drop(columns=["_ride_in_flag", "_ride_out_flag"])


def process_one_day(
    day: datetime,
    bike_all: pd.DataFrame,
    stations: pd.DataFrame,
    flow_dir: Path,
    out_root: Path,
    buffers: list[float],
) -> None:
    """Run the full pipeline for one day and write one CSV per buffer radius."""
    day_tag = day.strftime("%Y%m%d")

    flow_path = flow_dir / f"HIST_STO_5MFLOW_5M_{day_tag}.CSV"
    if not flow_path.exists():
        print(f"[skip] missing AFC file: {flow_path}")
        return

    flow_5min = pd.read_csv(flow_path, encoding="utf-8")
    for col in ("FSTR_LINEID", "FSTR_STATIONID"):
        flow_5min[col] = flow_5min[col].astype(str).str.strip()

    chain = reconstruct_travel_chain(bike_all, day)

    for buffer_m in buffers:
        pairs = match_bikes_to_stations(chain, stations, buffer_m)
        panel = aggregate_to_afc_grid(pairs, flow_5min.copy())

        out_dir = out_root / f"{int(buffer_m)}m"
        out_dir.mkdir(parents=True, exist_ok=True)
        panel.to_csv(out_dir / f"merged_{day_tag}.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bike-csv", type=Path, required=True)
    parser.add_argument("--station-coords", type=Path, required=True)
    parser.add_argument("--station-info", type=Path, required=True)
    parser.add_argument("--flow-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("outputs/panel"))
    parser.add_argument("--start", default="2022-12-12")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument(
        "--buffers",
        type=float,
        nargs="+",
        default=[50.0, 100.0, 200.0, 500.0],
        help="Manhattan buffer radii in metres.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stations = load_static_tables(args.station_coords, args.station_info)
    bike_all = pd.read_csv(args.bike_csv)

    days = pd.date_range(args.start, args.end, freq="D")
    for day in tqdm(days, desc="Processing days", unit="day"):
        process_one_day(
            day=day,
            bike_all=bike_all,
            stations=stations,
            flow_dir=args.flow_dir,
            out_root=args.out_root,
            buffers=args.buffers,
        )

    print(f"Done. Station-time panels written to {args.out_root}")


if __name__ == "__main__":
    main()
