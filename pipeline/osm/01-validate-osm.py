"""Run the OSM validation checks for the toml-specified OSM file.

1. Read PBF
2. Count IDs by feature type.
3. Return IDS by feature type.
4. Inspect tags by feature type.
"""

import toml
from pyprojroot import here
from transport_performance.osm.validate_osm import (
    FindIds,
    FindTags,
    FindLocations,
)
import time

CONFIG = toml.load(here("pipeline/osm/config/01-validate-osm.toml"))
OSM_PTH = CONFIG["OSM"]["PATH"]

pre_ids = time.perf_counter()
ids = FindIds(osm_pth=OSM_PTH)
ids.count_features()
ids.get_feature_ids()
ids.id_dict["way_ids"][0:11]
post_ids = time.perf_counter()
print(f"ID handling in {post_ids - pre_ids:0.4f} seconds")
# for fixture, ~0.9805 seconds
# for wales-latest ~46.8345 seconds

tags = FindTags(OSM_PTH)
# tags.node_tags[10971292662]  # subset by a single ID of interest
# tags.way_tags[1181392037]  # ways, relations and area tags are also available
# many nodes come without tags, this slice of available nodes shows how to
# return tags for found feature IDs, some with empty and some with populated
# tag lists.
tags.check_tags_for_ids(ids.id_dict["node_ids"][23:38], "node")
# get some way tags
tags.check_tags_for_ids(ids.id_dict["way_ids"][0:11], "way")
# get some relation tags
tags.check_tags_for_ids(ids.id_dict["relation_ids"][0:11], "relation")
# get some area tags
tags.check_tags_for_ids(ids.id_dict["area_ids"][0:11], "area")
post_tags = time.perf_counter()
print(f"tag handling in {post_tags - post_ids:0.4f} seconds")
# for fixture, ~11.4143 seconds
# for wales-latest ~515.9820 seconds -> 8mins.

# get node coordinates for node or way features:
locs = FindLocations(OSM_PTH)
# locs.node_locs[10971292664]  # get a known node ID's coord
# get a known way ID's member node coords
# locs.way_node_locs[1181392037]
# get the locs for a list of IDs. Firstly node features.
locs.check_locs_for_ids(ids.id_dict["node_ids"][0:11], feature_type="node")
# Also way features.
locs.check_locs_for_ids(ids.id_dict["way_ids"][0:11], feature_type="way")
post_locs = time.perf_counter()
print(f"location handling in {post_locs - post_tags:0.4f} seconds")
# for fixture, ~4.6656 seconds
# for wales-latest ~268.4524 seconds -> 4mins
