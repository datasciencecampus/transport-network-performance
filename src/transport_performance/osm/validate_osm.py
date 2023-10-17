"""Validation of OSM pbf files."""
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
    targets: dict, _list: list, feature_type: str
) -> dict:
    _type_defence(feature_type, "feature_type", str)
    _type_defence(targets, "targets", dict)
    _check_list(_list, "_list", exp_type=int)
    feat = feature_type.lower().strip()
    try:
        targ_dict = targets[feat]
    except KeyError:
        raise KeyError(
            f"`feature_type`: {feat} did not match keys in "
            f"`targets`: {targets.keys()}"
        )

    filtered_dict = dict(
        (id, targ_dict[id]) for id in _list if id in targ_dict
    )

    return {feat: filtered_dict}


# ---- look at internals
class GenericHandler(osmium.SimpleHandler):
    """Placeholder.

    Parameters
    ----------
    osmium : class
        Inherits from osmium.SimpleHandler

    """

    def __init__(self):
        super().__init__()
        self.node_ids = []
        self.node_locs = {}
        self.node_tags = {}
        self.way_ids = []
        self.way_nodes = {}
        self.way_tags = {}
        self.relations_ids = []
        self.relations_members = {}
        self.relations_tags = {}
        self.area_ids = []
        self.area_tags = {}

    # the methods you define must be called node, way, relation, area or
    # changeset

    def way(self, w):
        """Process ways.

        Parameters
        ----------
        w : osmium.osm.types.Way
            A 'way' feature.

        """
        self.way_ids.append(w.id)
        # compile the member nodes of each way
        nodelist = []
        for node in w.nodes:
            nodelist.append(node.ref)
        self.way_nodes[w.id] = nodelist
        # compile tags for each way
        tags_dict = _compile_tags(w)
        self.way_tags[w.id] = tags_dict

    def relation(self, r):
        """Process relations.

        Parameters
        ----------
        r : osmium.osm.types.Relation
            A 'relation' feature.

        """
        self.relations_ids.append(r.id)
        members_list = []
        # compile the relation members
        for member in r.members:
            members_list.append(member)
        self.relations_members[r.id] = members_list
        # compile the relation tags
        tags_dict = _compile_tags(r)
        self.relations_tags[r.id] = tags_dict

    def node(self, n):
        """Process nodes.

        Parameters
        ----------
        n : osmium.osm.types.Node
            A 'node' feature.

        """
        self.node_ids.append(n.id)
        # extract x,y
        x, y = str(n.location).split("/")
        # store representative point for each node
        self.node_locs[n.id] = {"lon": x, "lat": y}
        # get tags for each node
        tagdict = _compile_tags(n)
        self.node_tags[n.id] = tagdict

    def area(self, a):
        """Process areas.

        Parameters
        ----------
        a : osmium.osm.types.Area
            An 'Area' feature.

        """
        self.area_ids.append(a.id)
        tagdict = _compile_tags(a)
        self.area_tags[a.id] = tagdict


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

        Raises
        ------
        ValueError
            `feature_type` is not one of "node", "way", "relation" or "area".
            `found_tags` is empty. No tags for the combination of `ids` and
            `feature_type` were found.
        TypeError
            `ids` is not a list.
            Elements of `ids` are not integer.
            `feature_type` is not a string.

        """
        # defence
        _check_list(ids, "ids", exp_type=int)
        _type_defence(feature_type, "feature_type", str)
        feature_type = feature_type.lower().strip()
        _check_item_in_list(
            feature_type, ["node", "way", "relation", "area"], "feature_type"
        )
        # return the tags for the appropriate feature_type
        self.found_tags = _filter_target_dict_with_list(
            targets={
                "node": self.node_tags,
                "way": self.way_tags,
                "relation": self.relation_tags,
                "area": self.area_tags,
            },
            _list=ids,
            feature_type=feature_type,
        )

        if len(self.found_tags[feature_type]) == 0:
            raise ValueError(
                "No tags found. Did you specify the correct feature_type?"
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

        Raises
        ------
        ValueError
            `feature_type` is not one of "node" or "way".
            `found_locs` is empty. No coords for the combination of `ids` and
            `feature_type` were found.
        TypeError
            `ids` is not a list.
            Elements of `ids` are not integer.
            `feature_type` is not a string.

        """
        _type_defence(feature_type, "feature_type", str)
        feature_type = feature_type.lower().strip()
        _check_item_in_list(feature_type, ["node", "way"], "feature_type")
        _check_list(ids, "ids", exp_type=int)
        # return the filtered dict of locations
        self.found_locs = _filter_target_dict_with_list(
            targets={"node": self.node_locs, "way": self.way_node_locs},
            _list=ids,
            feature_type=feature_type,
        )

        if len(self.found_locs[feature_type]) == 0:
            raise ValueError(
                "No tags found. Did you specify the correct feature_type?"
            )

        return self.found_locs
