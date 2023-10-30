"""Visualise OMS spatial features."""
from pyprojroot import here
import pandas as pd
import geopandas as gpd
from shapely import Point
from transport_performance.osm import validate_osm as osmval

# %%
FIX_PTH = here("tests/data/newport-2023-06-13.osm.pbf")

# %%
# get some Node IDs to work with

nodes = osmval.FindIds(FIX_PTH)
nodes.get_feature_ids()
some_IDs = nodes.id_dict["node_ids"][0:100]

# %%
# get the locations
locs = osmval.FindLocations(FIX_PTH)
some_coords = locs.check_locs_for_ids(some_IDs, "node")
# %%
# get the contextual information in the tags, though note these
# be blank for nodes
# tags = osmval.FindTags(FIX_PTH)
# some_tags = tags.check_tags_for_ids(some_IDs, "node")
# all of the found tags are blank. TODO: find a populated example.

# %%
all_coords = pd.DataFrame()
for key, values in some_coords["node"].items():
    coord_row = pd.DataFrame(values, index=[key])
    all_coords = pd.concat([all_coords, coord_row])

# %%
all_coords["geometry"] = [
    Point(xy) for xy in zip(all_coords["lon"], all_coords["lat"])
]

# %%
coords_gdf = gpd.GeoDataFrame(all_coords, crs=4326)
coords_gdf.explore()

# %%
# %%
