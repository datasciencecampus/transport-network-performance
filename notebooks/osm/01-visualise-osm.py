"""Visualise OMS spatial features."""
# %%
from pyprojroot import here
from transport_performance.osm import validate_osm as osmval

# %%
fix_pth = here("tests/data/newport-2023-06-13.osm.pbf")

# %%
# get some Node IDs to work with
ids = osmval.FindIds(fix_pth)
ids.get_feature_ids()
node_ids = ids.id_dict["node_ids"][0:1000]
# get a way ID to work with
way_ids = ids.id_dict["way_ids"][0:1]

# %%
# find all location data in the fixture file
locs = osmval.FindLocations(fix_pth)
# %%
# plot the subset of node IDs
locs.plot_ids(ids=node_ids, feature_type="node")

# %%
# plot the selected way ID
locs.plot_ids(ids=way_ids, feature_type="way")

# %%
