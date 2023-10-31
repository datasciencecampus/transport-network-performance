"""Visualise OMS spatial features."""
# %%
from pyprojroot import here
import pandas as pd
import geopandas as gpd
from shapely import Point
from typing import Union
from transport_performance.osm import validate_osm as osmval

# %%
FIX_PTH = here("tests/data/newport-2023-06-13.osm.pbf")

# %%
# get some Node IDs to work with
ids = osmval.FindIds(FIX_PTH)
ids.get_feature_ids()
node_IDs = ids.id_dict["node_ids"][0:1000]

# %%
# get the locations
locs = osmval.FindLocations(FIX_PTH)
node_coords = locs.check_locs_for_ids(node_IDs, "node")["node"]
# %%
# get the contextual information in the tags, though note these
# be blank for nodes
# tags = osmval.FindTags(FIX_PTH)
# some_tags = tags.check_tags_for_ids(some_IDs, "node")
# all of the found tags are blank. TODO: find a populated example.

# %%


def _convert_osmdict_to_gdf(
    _dict: dict,
    feature_type: str = "node",
    lon_colnm: str = "lon",
    lat_colnm: str = "lat",
    _crs: Union[str, int] = "epsg:4326",
):
    """Convert an OSM dictionary to a GDF.

    Parameters
    ----------
    _dict: (dict)
        Dictionary of ID: location / tags.
    feature_type: str
        The type of feature data contained in the dictionary. Defaults to
        "node".
    lon_colnm: (str)
        Name of the longitude column.
    lat_colnm: (str)
        Name of the latitude column.
    _crs: (Union[str, int], optional)
        The CRS of the spatial features. Defaults to "epsg:4326".

    Returns
    -------
    out_gdf: (gpd.GeoDataFrame)
        A GeoDataFrame of the spatial features with ID mapped to index and
        values mapped to columns.

    """
    out_gdf = pd.DataFrame()
    out_row = pd.DataFrame()
    for key, values in _dict.items():
        if feature_type == "node":
            out_row = pd.DataFrame(values, index=[key])
        elif feature_type == "way":
            # now we have a nested list of node dictionaries. These are node
            # members of the way.
            for i in values:
                for k, j in i.items():
                    mem_row = pd.DataFrame(j, index=[key, k])
                    # if there are members of the parent ID, return multiindex
                    ind = pd.MultiIndex.from_tuples(
                        [(key, k)], names=["parent_id", "member_id"]
                    )
                    mem_row = pd.DataFrame(j, index=ind)
                    out_row = pd.concat([out_row, mem_row])
        # TODO - this line is behaving differently for feature type node versus
        # way. needs figuring out.
        out_gdf = pd.concat([out_gdf, out_row])

    out_gdf["geometry"] = [
        Point(xy) for xy in zip(out_gdf[lon_colnm], out_gdf[lat_colnm])
    ]
    out_gdf = gpd.GeoDataFrame(out_gdf, crs=_crs)
    return out_gdf


# %%
nodes_gdf = _convert_osmdict_to_gdf(node_coords)
nodes_gdf.explore()


# %%
way_ids = ids.id_dict["way_ids"][0:400]
way_coords = locs.check_locs_for_ids(way_ids, "way")["way"]
# %%
ways_gdf = _convert_osmdict_to_gdf(way_coords, "way", "x", "y")
ways_gdf.explore()
