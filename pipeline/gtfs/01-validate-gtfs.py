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

from heimdall_transport.gtfs.validation import Gtfs_Instance

CONFIG = toml.load(here("pipeline/gtfs/config/01-validate-gtfs.toml"))
GTFS_PTH = here(CONFIG["GTFS"]["PATH"])
UNITS = CONFIG["GTFS"]["UNITS"]
GEOM_CRS = CONFIG["GTFS"]["GEOMETRIC_CRS"]
POINT_MAP_PTH = CONFIG["MAPS"]["STOP_COORD_PTH"]
HULL_MAP_PATH = CONFIG["MAPS"]["STOP_HULL_PTH"]

feed = Gtfs_Instance(gtfs_pth=GTFS_PTH, units=UNITS)
print(feed.is_valid())
print(feed.validity_df["type"].value_counts())
feed.print_alerts()
feed.print_alerts(alert_type="warning")
feed.clean_feed()
print(feed.is_valid())
print(feed.validity_df["type"].value_counts())
feed.print_alerts()
feed.print_alerts(alert_type="warning")
# visualise gtfs
feed.viz_stops(out_pth=POINT_MAP_PTH)
print(f"Map written to {POINT_MAP_PTH}")
feed.viz_stops(out_pth=HULL_MAP_PATH, geoms="hull", geom_crs=GEOM_CRS)
print(f"Map written to {HULL_MAP_PATH}")
print(feed.get_route_modes())
with warnings.catch_warnings():  # slow & triggers warnings, gtfs_kit issue
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    print(feed.summarise_weekday())
