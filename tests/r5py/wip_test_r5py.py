"""Unit testing for 45py consistency (WIP)."""
# %%
import datetime as dt
import os
import sys

import geopandas as gpd
from pyprojroot import here

# import shapely
import pandas as pd

# allow 8GB of memory to java virtual instance - must be before importing r5py
sys.argv.append(["--max-memory", "8G"])
import r5py  # noqa

# %%
path_pbf = os.path.join(here(), "tests", "data", "newport-2023-06-13.osm.pbf")
path_gtfs = os.path.join(here(), "tests", "data", "newport-20230613_gtfs.zip")
centroids = os.path.join(here(), "tests", "data", "newport_centroids.pkl")

# %%
centroids_df = pd.read_pickle(centroids)
centroids = gpd.GeoDataFrame(
    centroids_df, geometry="centroid", crs=centroids_df.crs
)

# %%
transport_network = r5py.TransportNetwork(path_pbf, [path_gtfs])

# %%
# %%
travel_time_matrix_computer = r5py.TravelTimeMatrixComputer(
    transport_network,
    origins=centroids,
    destinations=centroids,
    departure=dt.datetime(2023, 6, 13, 8, 0),
    transport_modes=[r5py.TransportMode.TRANSIT],
)
travel_time_matrix = travel_time_matrix_computer.compute_travel_times()

median_times = travel_time_matrix.groupby("from_id")["travel_time"].median()

join = centroids.merge(median_times, left_on="id", right_on="from_id")
join.head()
join.explore("travel_time", cmap="Greens", marker_kwds={"radius": 12})
# %%
