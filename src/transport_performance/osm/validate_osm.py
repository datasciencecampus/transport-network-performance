"""Validation of OSM pbf files.

pyosmium requires handler classes that inherit from their API. The methods of
these handlers must be named `node`, `way`, `relation` or `area`. Exposing
these methods would be of limited utility, therefore internal handler classes
are used to collect and process the feature information.

A separate group of classes are then defined, inheriting from these internal
handlers. These API classes are used to apply the logic of the handler classes
to an osm.pbf file. The API classes also define methods associated with the
user requirements, eg 'find way Ids', 'find the coordinates for this list of
node IDs' or similar.

The API classes expose logic that enables users to:
* Count all features in a pbf file by type (node, way, relation or area)
* Return IDs of the above feature types
* Extract tags for the above features
* Find coordinates for node or way features
* Plot the coordinates of a given list of node or way IDs
"""
import osmium
from pathlib import Path
from typing import Union
import pandas as pd
import geopandas as gpd
from shapely import Point
import folium

from transport_performance.utils.defence import (
    _check_item_in_iter,
    _check_iterable,
    _type_defence,
    _is_expected_filetype,
)

# ---------utilities-----------


def _compile_tags(osmium_feature):
    """Return tag name value pairs.

    Parameters
    ----------
    osmium_feature : osmium.osm.types
        Either a node, way, relation or area osmium type.

    Returns
    -------
    tagdict: dict
        A dictionary of tag name & values.

    """
    return {tag.k: tag.v for tag in osmium_feature.tags}


def _filter_target_dict_with_list(
    targets: dict, _list: list, search_key: str, accepted_keys: list
) -> dict:
    """Select a target dictionary and filter it with a list of keys.

    Parameters
    ----------
    targets : dict
        A dictionary of feature_type: attribute dicts.
    _list : list
        A list of keys, OSM IDs in this context. Integer keys are expected.
    search_key : str
        The key value to search with.
    accepted_keys : list
        Valid key values specific to the context of the method calling this
        function.

    Returns
    -------
    filtered_dict.
        Dictionary with keys filtered to IDs available in `_list`.

    Raises
    ------
    ValueError
        `search_key` is not in `accepted_keys`.
        `filtered_dict` is empty. No tags for the combination of `_list` and
        `search_key` were found.
    TypeError
        `_list` is not a list.
        `accepted_keys` is not a list.
        Elements of `_list` are not integer.
        Elements of `accepted_keys` are not string.
        `search_key` is not a string.
    KeyError
        `search_key` is not present in `targets.keys()`

    """
    _type_defence(targets, "targets", dict)
    _check_iterable(_list, "_list", list, exp_type=int)
    _type_defence(search_key, "search_key", str)
    _check_iterable(accepted_keys, "accepted_keys", list, exp_type=str)
    _check_item_in_iter(search_key, accepted_keys, "search_key")

    feat = search_key.lower().strip()
    try:
        targ_dict = targets[feat]
    except KeyError:
        raise KeyError(
            f"`search_key`: {feat} did not match keys in "
            f"`targets`: {targets.keys()}"
        )

    filtered_dict = dict(
        (id, targ_dict[id]) for id in _list if id in targ_dict
    )
    if len(filtered_dict) == 0:
        raise ValueError(
            "No tags found. Did you specify the correct search_key?"
        )

    return filtered_dict


def _convert_osm_dict_to_gdf(
    osm_dict: dict,
    feature_type: str = "node",
    crs: Union[str, int] = "epsg:4326",
) -> gpd.GeoDataFrame:
    """Convert an OSM dictionary to a GDF.

    Parameters
    ----------
    osm_dict: (dict)
        Dictionary of ID: location / tags.
    feature_type: str
        The type of feature data contained in the dictionary. Defaults to
        "node".
    crs: (Union[str, int], optional)
        The CRS of the spatial features. Defaults to "epsg:4326".

    Returns
    -------
    out_gdf: (gpd.GeoDataFrame)
        A GeoDataFrame of the spatial features with ID mapped to index and
        values mapped to columns.

    """
    out_df = pd.DataFrame()
    for key, values in osm_dict.items():
        if feature_type == "node":
            out_df = pd.concat([out_df, pd.DataFrame(values, index=[key])])
        elif feature_type == "way":
            # now we have a nested list of node dictionaries. These are node
            # members of the way.
            for i in values:
                for k, j in i.items():
                    mem_row = pd.DataFrame(j, index=[key, k])
                    # parent id is way, member id is node
                    ind = pd.MultiIndex.from_tuples(
                        [(key, k)], names=["parent_id", "member_id"]
                    )
                    mem_row = pd.DataFrame(j, index=ind)
                    out_df = pd.concat([out_df, mem_row])

    # geodataframe requires geometry column
    out_df["geometry"] = [
        Point(xy) for xy in zip(out_df["lon"], out_df["lat"])
    ]
    out_gdf = gpd.GeoDataFrame(out_df, crs=crs)
    return out_gdf


# ---------Internal Classes-----------


class _IdHandler(osmium.SimpleHandler):
    """Collate Ids for OSM Features.

    Internal class, method names must be fixed to integrate with pyosmium.

    Parameters
    ----------
    osmium.SimpleHandler : class
        Inherits from osmium.SimpleHandler

    Methods
    -------
    node()
        Collates available OSM node feature IDs. Creates the node_ids
        attribute.
    way()
        Collates available OSM way feature IDs. Creates the way_ids attribute.
    relation()
        Collates available OSM relation feature IDs. Creates the relations_ids
        attribute.
    area()
        Collates available OSM area feature IDs. Creates the area_ids attribute
        .

    Attributes
    ----------
    node_ids: list
        List of available OSM node feature IDs.
    way_ids: list
        List of available OSM way feature IDs.
    relations_ids: list
        List of available OSM relation feature IDs.
    area_ids: list
        List of available OSM area feature IDs.

    """

    def __init__(self) -> None:
        super().__init__()
        self.node_ids = list()
        self.way_ids = list()
        self.relations_ids = list()
        self.area_ids = list()

    def node(self, n: osmium.Node) -> None:
        """Collate node IDs.

        Parameters
        ----------
        n : osmium.Node
            A node feature.

        """
        self.node_ids.append(n.id)

    def way(self, w: osmium.Way) -> None:
        """Collate way IDs.

        Parameters
        ----------
        w : osmium.Way
            A way feature.

        """
        self.way_ids.append(w.id)

    def relation(self, r: osmium.Relation) -> None:
        """Collate relation IDs.

        Parameters
        ----------
        r : osmium.Relation
            A relation feature.

        """
        self.relations_ids.append(r.id)

    def area(self, a: osmium.Area) -> None:
        """Collate area IDs.

        Parameters
        ----------
        a : osmium.Area
            An area feature (includes boundaries).

        """
        self.area_ids.append(a.id)


class _TagHandler(osmium.SimpleHandler):
    """Collate tags for OSM features.

    Parameters
    ----------
    osmium.SimpleHandler : class
        Inherits from osmium.SimpleHandler

    Methods
    -------
    node()
        Compiles all available tag data for OSM node features. Creates the
        node_tags attribute.
    way()
        Compiles all available tag data for OSM way features. Creates the
        way_tags attribute.
    relation()
        Compiles all available tag data for OSM relation features. Creates the
        relation_tags attribute.
    area()
        Compiles all available tag data for OSM area features. Creates the
        area_tags attribute.

    """

    def __init__(self) -> None:
        super().__init__()
        self.node_tags = dict()
        self.way_tags = dict()
        self.relation_tags = dict()
        self.area_tags = dict()

    def node(self, n: osmium.Node) -> None:
        """Collate node tags.

        Parameters
        ----------
        n : osmium.Node
            A node feature.

        """
        # get tags for each node
        tagdict = _compile_tags(n)
        self.node_tags[n.id] = tagdict

    def way(self, w: osmium.Way) -> None:
        """Collate way tags.

        Parameters
        ----------
        w : osmium.Way
            A way feature.

        """
        # get tags for each way
        tagdict = _compile_tags(w)
        self.way_tags[w.id] = tagdict

    def relation(self, r: osmium.Relation) -> None:
        """Collate relation tags.

        Parameters
        ----------
        r : osmium.Relation
            A relation feature.

        """
        # get tags for each relation
        tagdict = _compile_tags(r)
        self.relation_tags[r.id] = tagdict

    def area(self, a: osmium.Area) -> None:
        """Collate area tags.

        Parameters
        ----------
        a : osmium.Area
            An area feature (includes boundaries).

        """
        # get tags for each area
        tagdict = _compile_tags(a)
        self.area_tags[a.id] = tagdict


class _LocHandler(osmium.SimpleHandler):
    """Collate coordinates for nodes or ways.

    Parameters
    ----------
    osmium.SimpleHandler : class
        Inherits from osmium.SimpleHandler

    """

    def __init__(self) -> None:
        super().__init__()
        self.node_locs = dict()
        self.way_node_locs = dict()

    def node(self, n):
        """Collate node coordinates.

        Parameters
        ----------
        n : osmium.Node
            A node feature.

        """
        # extract x,y
        x, y = str(n.location).split("/")
        # store representative point for each node
        self.node_locs[n.id] = {"lon": float(x), "lat": float(y)}

    def way(self, w):
        """Collate coordinates for member nodes of a way.

        Parameters
        ----------
        w : osmium.Way
            A way feature.

        """
        # compile the member nodes of each way
        nodelist = []
        for node in w.nodes:
            id, coords = str(node).split("@")
            coords = coords.replace("]", "")
            x, y = coords.split("/")
            nodelist.append({node.ref: {"lon": float(x), "lat": float(y)}})
        self.way_node_locs[w.id] = nodelist


# ---------API classes-----------


class FindIds:
    """Apply ID collation to an OSM file.

    Count or return available feature IDs in an OSM file.

    Parameters
    ----------
    osm_pth: Union[Path, str]
        Path to osm file.
    id_collator: _IdHandler, optional
        FindIds applies the logic from id_collator to a pbf file on init,
        storing the collated IDs in their appropriate attributes - __node_ids,
        __way_ids, __relations_ids and __area_ids. Defaults to _IdHandler.

    Raises
    ------
    TypeError:
        `osm_pth` is not of type pathlib.Path or str.
    FileNotFoundError:
        `osm_pth` file not found on disk.
    ValueError:
        `osm_pth` does not have a .pbf extension.

    Attributes
    ----------
    counts: dict
        Counts of feature IDs by feature type.
    id_dict: dict
        IDs of all found features by feature type.
    __node_ids: list
        Internal attribute contains list of all node IDs contained in pbf file.
    __way_ids: list
        Internal attribute contains list of all way IDs contained in pbf file.
    __relations_ids: list
        Internal attribute contains list of all relation IDs contained in pbf
        file.
    __area_ids: list
        Internal attribute contains list of all area IDs contained in pbf file.

    Methods
    -------
    count_features()
        Count of feature IDs by feature type.
    get_feature_ids()
        Return feature IDs by available feature type.

    """

    def __init__(
        self, osm_pth: Union[Path, str], id_collator: _IdHandler = _IdHandler
    ) -> None:
        _is_expected_filetype(
            osm_pth, "osm_pth", check_existing=True, exp_ext=".pbf"
        )
        collated_ids = id_collator()
        collated_ids.apply_file(osm_pth)
        self.__node_ids = collated_ids.node_ids
        self.__way_ids = collated_ids.way_ids
        self.__relations_ids = collated_ids.relations_ids
        self.__area_ids = collated_ids.area_ids
        self.counts = dict()
        self.id_dict = dict()

    def count_features(self) -> dict:
        """Count numbers of each available feature type.

        Returns
        -------
        counts: dict
            Counts of node, way, relation & area IDs in a pbf file.

        """
        counts = {
            "n_nodes": len(self.__node_ids),
            "n_ways": len(self.__way_ids),
            "n_relations": len(self.__relations_ids),
            "n_areas": len(self.__area_ids),
        }
        self.counts = counts
        return counts

    def get_feature_ids(self) -> dict:
        """Get a list of all available IDs in a pbf file for each feature type.

        Returns
        -------
        id_dict: dict
            Available IDs for nodes, ways, relations and areas.

        """
        id_dict = {
            "node_ids": self.__node_ids,
            "way_ids": self.__way_ids,
            "relation_ids": self.__relations_ids,
            "area_ids": self.__area_ids,
        }
        self.id_dict = id_dict
        return id_dict


class FindTags:
    """Applies tag collation to OSM file.

    Parameters
    ----------
    osm_pth: Union[Path, str]
        Path to osm file.
    tag_collator: _TagHandler, optional
        FindTags applies the logic from tag_collator to a pbf file on init,
        storing the collated tags in a `found_tags` attribute. Defaults to
        _TagHandler.

    Raises
    ------
    TypeError:
        `osm_pth` is not of type pathlib.Path or str.
    FileNotFoundError:
        `osm_pth` file not found on disk.
    ValueError:
        `osm_pth` does not have a .pbf extension.

    Attributes
    ----------
    found_tags: dict
        Found tags for specified feature IDs.
    __node_tags: dict
        Tags found for OSM node features.
    __way_tags: dict
        Tags found for OSM way features.
    __relation_tags: dict
        Tags found for OSM relation features.
    __area_tags: dict
        Tags found for OSM area features.

    Methods
    -------
    check_tags_for_ids()
        Filter tags to the given list of IDs. Updates `found_tags` attribute.

    """

    def __init__(
        self, osm_pth, tag_collator: _TagHandler = _TagHandler
    ) -> None:
        _is_expected_filetype(
            osm_pth, "osm_pth", check_existing=True, exp_ext=".pbf"
        )
        tags = _TagHandler()
        tags.apply_file(osm_pth)
        self.__node_tags = tags.node_tags
        self.__way_tags = tags.way_tags
        self.__relation_tags = tags.relation_tags
        self.__area_tags = tags.area_tags
        self.found_tags = dict()

    def check_tags_for_ids(self, ids: list, feature_type: str) -> dict:
        """Return tags for provided list of feature IDs.

        Parameters
        ----------
        ids : list
            A list of OSM feature IDs to check. IDs must be integer.
        feature_type : str
            The type of feature to which the IDS belong. Valid options are
            "node", "way", "relation", or "area".

        Returns
        -------
        found_tags: dict
            ID: dict of tags, containing tag name : value.

        """
        self.found_tags = _filter_target_dict_with_list(
            targets={
                "node": self._FindTags__node_tags,
                "way": self._FindTags__way_tags,
                "relation": self._FindTags__relation_tags,
                "area": self._FindTags__area_tags,
            },
            _list=ids,
            search_key=feature_type,
            accepted_keys=["node", "way", "relation", "area"],
        )
        return self.found_tags


class FindLocations(_LocHandler):
    """Applies location collation to OSM file.

    Parameters
    ----------
    _LocHandler : class
        Internal class for handling locations. Inherits from
        osmium.SimpleHandler.
    osm_pth: Union[Path, str]
        Path to osm file.

    Raises
    ------
    TypeError:
        `osm_pth` is not of type pathlib.Path or str.
    FileNotFoundError:
        `osm_pth` file not found on disk.
    ValueError:
        `osm_pth` does not have a .pbf extension.

    Attributes
    ----------
    found_locs: dict
        Found locations for specified feature IDs.
    node_locs: dict
        Node coordinates. Inherited from class _LocHandler.
    way_node_locs: dict
        Member node coordinates for each way feature. Inherited from class
        _LocHandler.

    Methods
    -------
    check_locs_for_ids()
        Filter locations to the given list of IDs. Updates `found_locs`
        attribute.
    node()
        Gets coordinate data from a node. Creates the `node_locs` attribute.
        Inherited from class _LocHandler.
    way()
        Gets coordinate data for each node member of a way. Creates the
        `way_node_locs` attribute. Inherited from class _LocHandler.

    """

    def __init__(self, osm_pth) -> None:
        super().__init__()
        _is_expected_filetype(osm_pth, "osm_pth", exp_ext=".pbf")
        self.apply_file(osm_pth, locations=True)
        self.found_locs = dict()

    def check_locs_for_ids(self, ids: list, feature_type: str) -> dict:
        """Return coordinates for provided list of feature IDs.

        Parameters
        ----------
        ids : list
            A list of OSM feature IDs to check. IDs must be integer.
        feature_type : str
            The type of feature to which the IDS belong. Valid options are
            "node" or "way".

        Returns
        -------
        found_locs: dict
            ID: dict of tags, containing ID : location.

        """
        self.found_locs = _filter_target_dict_with_list(
            targets={"node": self.node_locs, "way": self.way_node_locs},
            _list=ids,
            search_key=feature_type,
            accepted_keys=["node", "way"],
        )
        return self.found_locs

    def plot_ids(
        self,
        ids: list,
        feature_type: str,
        crs: Union[str, int] = "epsg:4326",
    ) -> folium.Map:
        """Plot coordinates for nodes or node members of a way.

        Provided with a list of node or way IDs, converts the coordinate data
        from dictionary to GeoDataFrame and uses the basic gdf.explore() method
        to visualise the features on a basemap.

        Parameters
        ----------
        ids : list
            A list of Node or Way IDs.
        feature_type : str
            Whether the type of OSM feature to plot is node or way.
        crs : Union[str, int], optional
            The projection of the spatial features, by default "epsg:4326"

        Returns
        -------
        folium.Map
            A plot of the coordinate data for each identifiable node.

        Raises
        ------
        NotImplementedError
            Relation location data could be extracted with a bit more munging.
            Please raise a feature request if you feel this is significant.
        ValueError
            `feature_type` is not one of "node", "way", "relation" or "area".
        TypeError
            `ids` is not of type list.
            `feature_type` is not of type str.

        """
        _type_defence(ids, "ids", list)
        _type_defence(feature_type, "feature_type", str)
        _type_defence(crs, "crs", (str, int))
        feature_type = feature_type.lower().strip()
        ACCEPT_FEATS = ["node", "way", "relation", "area"]
        _check_item_in_iter(
            item=feature_type, iterable=ACCEPT_FEATS, param_nm="feature_type"
        )
        if feature_type != "node" and feature_type != "way":
            raise NotImplementedError(
                "Relation or area plotting not implemented at this time."
            )
        self.check_locs_for_ids(ids, feature_type)
        self.coord_gdf = _convert_osm_dict_to_gdf(
            osm_dict=self.found_locs,
            feature_type=feature_type,
            crs=crs,
        )
        return self.coord_gdf.explore()
