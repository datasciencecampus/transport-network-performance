"""Validation of OSM pbf files.

pyosmium requires handler classes that inherit from their API. The methods of
these handlers must be named `node`, `way`, `relation` or `area`. Exposing
these methods would offer no functionality, therefore internal handler classes
are used to collect and process the feature information.

A separate group of classes are then defined, inheriting from these internal
handlers. These API classes are used to apply the logic of the handler classes
to an osm.pbf file. The API classes also define methods associated with the
user requirements, eg 'find way Ids', 'find the coordinates for this list of
node IDs' or similar.
"""
import osmium
import warnings
from typing import Any
from pathlib import Path
from typing import Union

from transport_performance.utils.defence import (
    _check_item_in_list,
    _check_list,
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
    tagdict = dict()
    for tag in osmium_feature.tags:
        # taglist.append(tag)
        for i, tag in enumerate(tag):
            if i % 2 == 0:
                # even tags are keys
                k = tag
            else:
                # odd tags are values
                v = tag
        tagdict[k] = v
    return tagdict


def _check_dict_values_all_equal(a_dict: dict, a_value: Any) -> bool:
    """Check if all dict values equal a_value.

    Parameters
    ----------
    a_dict : dict
        A dictionary.
    a_value : Any
        A value to check the dictionary values against.

    Returns
    -------
    bool
        True if all dictionary values equals `a_value`, else False.

    """
    return all([i == a_value for i in a_dict.values()])


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
    dict
        search_key: filtered_dict. Dictionary with keys filtered to IDs
        available in `_list`.

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
    _type_defence(search_key, "feature_type", str)
    _check_list(_list, "_list", exp_type=int)
    _check_list(accepted_keys, "accepted_features", exp_type=str)
    _check_item_in_list(search_key, accepted_keys, "search_key")

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

    return {feat: filtered_dict}


# ---------Internal Classes-----------


class _IdHandler(osmium.SimpleHandler):
    """Collate Ids for OSM Features.

    Internal class, method names must be fixed to integrate with pyosmium.

    Parameters
    ----------
    osmium.SimpleHandler : class
        Inherits from osmium.SimpleHandler

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
        self.node_locs[n.id] = {"lon": x, "lat": y}

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
            nodelist.append({node.ref: {"x": float(x), "y": float(y)}})
        self.way_node_locs[w.id] = nodelist


# ---------API classes-----------


class FindIds(_IdHandler):
    """Apply ID collation to an OSM file.

    Count or return available feature IDs in an OSM file.

    Parameters
    ----------
    _IdHandler : class
        Internal class for handling IDs. Inherits from osmium.SimpleHandler.
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
    counts: dict
        Counts of feature IDs by feature type.
    id_dict: dict
        IDs of all found features by feature type.
    node_ids: list
        List of available OSM node feature IDs. Inheroted from _IdHandler.
    way_ids: list
        List of available OSM way feature IDs. Inheroted from _IdHandler.
    relations_ids: list
        List of available OSM relation feature IDs. Inheroted from _IdHandler.
    area_ids: list
        List of available OSM area feature IDs. Inheroted from _IdHandler.

    Methods
    -------
    count_features()
        Count of feature IDs by feature type.
    get_feature_ids()
        Return feature IDs by available feature type.
    node()
        Collates available OSM node feature IDs. Creates the node_ids
        attribute. Inherited from _IdHandler.
    way()
        Collates available OSM way feature IDs. Creates the way_ids attribute.
        Inherited from _IdHandler.
    relation()
        Collates available OSM relation feature IDs. Creates the relations_ids
        attribute. Inherited from _IdHandler.
    area()
        Collates available OSM area feature IDs. Creates the area_ids
        attribute. Inherited from _IdHandler.

    """

    def __init__(self, osm_pth: Union[Path, str]) -> None:
        super().__init__()
        _is_expected_filetype(
            osm_pth, "osm_pth", check_existing=True, exp_ext=".pbf"
        )
        self.apply_file(osm_pth)
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
            "n_nodes": len(self.node_ids),
            "n_ways": len(self.way_ids),
            "n_relations": len(self.relations_ids),
            "n_areas": len(self.area_ids),
        }
        # in cases where the user has not ran the apply_file method, warn:
        if _check_dict_values_all_equal(counts, 0):
            warnings.warn(
                "No counts were found, did you run `self.apply_file"
                "(<INSERT PBF PATH>)`?",
                UserWarning,
            )
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
            "node_ids": self.node_ids,
            "way_ids": self.way_ids,
            "relation_ids": self.relations_ids,
            "area_ids": self.area_ids,
        }
        if _check_dict_values_all_equal(id_dict, []):
            warnings.warn(
                "No Ids were found. Did you run "
                "`self.apply_file(<INSERT PBF PATH>)?`",
                UserWarning,
            )
        self.id_dict = id_dict
        return id_dict


class FindTags(_TagHandler):
    """Applies tag collation to OSM file.

    Parameters
    ----------
    _TagHandler : class
        Internal class for handling IDs. Inherits from osmium.SimpleHandler.
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
    found_tags: dict
        Found tags for specified feature IDs.
    node_tags: dict
        Tags found for OSM node features. Inherited from _TagHandler.
    way_tags: dict
        Tags found for OSM way features. Inherited from _TagHandler.
    relation_tags: dict
        Tags found for OSM relation features. Inherited from _TagHandler.
    area_tags: dict
        Tags found for OSM area features. Inherited from _TagHandler.

    Methods
    -------
    check_tags_for_ids()
        Filter tags to the given list of IDs. Updates `found_tags` attribute.
    node()
        Compiles all available tag data for OSM node features. Creates the
        node_tags attribute. Inherited from _TagHandler.
    way()
        Compiles all available tag data for OSM way features. Creates the
        way_tags attribute. Inherited from _TagHandler.
    relation()
        Compiles all available tag data for OSM relation features. Creates the
        relation_tags attribute. Inherited from _TagHandler.
    area()
        Compiles all available tag data for OSM area features. Creates the
        area_tags attribute. Inherited from _TagHandler.

    """

    def __init__(self, osm_pth) -> None:
        super().__init__()
        _is_expected_filetype(
            osm_pth, "osm_pth", check_existing=True, exp_ext=".pbf"
        )
        self.apply_file(osm_pth)
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
                "node": self.node_tags,
                "way": self.way_tags,
                "relation": self.relation_tags,
                "area": self.area_tags,
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

    def check_locs_for_ids(self, ids: list, feature_type: str):
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
