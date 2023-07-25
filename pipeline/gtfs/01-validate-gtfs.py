"""Run the GTFS validation checks for the toml-specified GTFS file.

1. read feed
2. describe feed
3. validate feed
4. clean feed
5. new  - print errors / warnings in full
6. new - visualise convex hull of stops and area
7. visualise stop locations
8. new - modalities available (including extended spec)
9. new - feed stats by is-weekend
"""
import toml
from pyprojroot import here
import warnings
import pandas as pd
import time
import subprocess

from transport_performance.gtfs.validation import GtfsInstance
from transport_performance.utils.defence import _is_expected_filetype

CONFIG = toml.load(here("pipeline/gtfs/config/01-validate-gtfs.toml"))
GTFS_PTH = here(CONFIG["GTFS"]["PATH"])
UNITS = CONFIG["GTFS"]["UNITS"]
GEOM_CRS = CONFIG["GTFS"]["GEOMETRIC_CRS"]
POINT_MAP_PTH = CONFIG["MAPS"]["STOP_COORD_PTH"]
HULL_MAP_PATH = CONFIG["MAPS"]["STOP_HULL_PTH"]
PROFILING = CONFIG["UTILS"]["PROFILING"]
# check GTFS Path exists
_is_expected_filetype(pth=GTFS_PTH, param_nm="GTFS_PTH", check_existing=True)
# Get the disk usage of the GTFS file.
gtfs_du = (
    subprocess.check_output(["du", "-sh", GTFS_PTH]).split()[0].decode("utf-8")
)
if PROFILING:
    print(f"GTFS at {GTFS_PTH} disk usage: {gtfs_du}")

pre_init = time.perf_counter()
feed = GtfsInstance(gtfs_pth=GTFS_PTH, units=UNITS)
post_init = time.perf_counter()
if PROFILING:
    print(f"Init in {post_init - pre_init:0.4f} seconds")

feed.get_calendar_dates()
post_dates = time.perf_counter()
if PROFILING:
    print(f"get_calendar_dates in {post_dates - post_init:0.4f} seconds")
s = feed.available_dates[0]
f = feed.available_dates[-1]
print(f"{len(feed.available_dates)} dates available between {s} & {f}.")

try:
    # If agency_id is missing, an AttributeError is raised. GTFS spec states
    # This is conditionally required, dependent if more than one agency is
    # operating within the feed. https://gtfs.org/schedule/reference/#agencytxt
    # Cleaning the feed doesn't resolve. Raise issue to investigate.
    print(feed.is_valid())
    post_isvalid = time.perf_counter()
    if PROFILING:
        print(f"is_valid in {post_isvalid - post_dates:0.4f} seconds")
    print(feed.validity_df["type"].value_counts())
    feed.print_alerts()
    post_errors = time.perf_counter()
    feed.print_alerts(alert_type="warning")
    post_warn = time.perf_counter()
    if PROFILING:
        print(f"print_alerts errors: {post_errors - post_isvalid:0.4f} secs")
        print(f"print_alerts warn: {post_warn - post_errors:0.4f} secs")
except AttributeError:
    print("AttributeError. Unable to validate feed.")

pre_clean = time.perf_counter()
feed.clean_feed()
post_clean = time.perf_counter()
if PROFILING:
    print(f"clean_feed in {post_clean - pre_clean:0.4f} seconds")

try:
    print(feed.is_valid())
    print(feed.validity_df["type"].value_counts())
    feed.print_alerts()
    feed.print_alerts(alert_type="warning")
except AttributeError:
    print("AttributeError. Unable to validate feed.")

# visualise gtfs
pre_viz_points = time.perf_counter()
feed.viz_stops(out_pth=POINT_MAP_PTH)
post_viz_points = time.perf_counter()
if PROFILING:
    print(f"viz_points in {post_viz_points - pre_viz_points:0.4f} seconds")
print(f"Map written to {POINT_MAP_PTH}")

pre_viz_hull = time.perf_counter()
feed.viz_stops(out_pth=HULL_MAP_PATH, geoms="hull", geom_crs=GEOM_CRS)
post_viz_hull = time.perf_counter()
if PROFILING:
    print(f"viz_hull in {post_viz_hull - pre_viz_hull:0.4f} seconds")
print(f"Map written to {HULL_MAP_PATH}")

pre_route_modes = time.perf_counter()
print(feed.get_route_modes())
post_route_modes = time.perf_counter()
if PROFILING:
    print(f"route_modes in {post_route_modes - pre_route_modes:0.4f} seconds")

pre_summ_weekday = time.perf_counter()
with warnings.catch_warnings():  # slow & triggers warnings, gtfs_kit issue
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    print(feed.summarise_weekday())
post_summ_weekday = time.perf_counter()
if PROFILING:
    print(f"summ_weekday in {post_summ_weekday - pre_summ_weekday:0.4f} secs")
    print(f"Pipeline execution in {post_summ_weekday - pre_init:0.4f}")
