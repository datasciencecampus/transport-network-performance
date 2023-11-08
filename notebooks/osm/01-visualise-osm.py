"""Visualise OMS spatial features."""
# %%
from pyprojroot import here
from transport_performance.osm import validate_osm as osmval
from transport_performance.osm.validate_osm import _convert_osmdict_to_gdf

# %%
fix_pth = here("tests/data/newport-2023-06-13.osm.pbf")

# %%
# get some Node IDs to work with
ids = osmval.FindIds(fix_pth)
ids.get_feature_ids()
node_IDs = ids.id_dict["node_ids"][0:1000]

# %%
# get the locations
locs = osmval.FindLocations(fix_pth)
node_coords = locs.check_locs_for_ids(node_IDs, "node")["node"]


# %%
nodes_gdf = _convert_osmdict_to_gdf(node_coords)
nodes_gdf.explore()


# %%
way_ids = ids.id_dict["way_ids"][0:1]
way_coords = locs.check_locs_for_ids(way_ids, "way")["way"]
# %%
ways_gdf = _convert_osmdict_to_gdf(way_coords, "way")
ways_gdf.explore()
